"""YOLOv8-nano wrapper for the agents flow.

Pure-PyTorch reimplementation of the ultralytics YOLOv8n architecture
that traces cleanly through torch.fx.symbolic_trace. The original
ultralytics C2f.forward uses Python list + generator extend, and the
top-level DetectionModel.forward is a Python loop over self.model
with conditional skip-connection routing — neither is fx-traceable.

This wrapper:
  * statically unrolls C2f into n=1 / n=2 variants (the only counts
    YOLOv8n uses);
  * wires backbone / neck / head as a flat sequence of submodule calls
    with explicit cat / upsample / chunk operations;
  * exposes only the raw 3-scale detection head outputs (post-NMS
    decode and DFL stay in PyTorch land — the C harness needs only
    the raw conv outputs).

`get_model()` loads pretrained ultralytics yolov8n.pt weights into the
matching submodules. The COCO-trained weights stream in cleanly since
backbone/neck channel counts are identical; head re-init to a smaller
`nc` would be a fine-tune-time concern, out of scope here.

Env knobs:
  AGENTS_YOLOV8N_INPUT       default 160 (must be a multiple of 32)
  AGENTS_YOLOV8N_NC          default 80 (COCO classes)
  AGENTS_YOLOV8N_PRETRAINED  default 1 (load yolov8n.pt; 0 → random init)
"""

from __future__ import annotations

import os
import re
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _Conv(nn.Module):
    """Conv2d + BatchNorm2d + SiLU. Matches ultralytics' Conv module."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1,
                 p: Optional[int] = None, g: int = 1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        # ultralytics' BN defaults — must match exactly for pretrained
        # weight numerics to round-trip. Standard pytorch uses 1e-5/0.1
        # which silently corrupts the activations by 1-2 orders of
        # magnitude per BN layer.
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _Bottleneck(nn.Module):
    """YOLOv8 bottleneck: two 3×3 Convs, optional residual.

    Always operates on c×H×W tensors (c1 == c2), so the residual
    add is dimensionally valid whenever shortcut=True.
    """

    def __init__(self, c: int, shortcut: bool = True):
        super().__init__()
        self.cv1 = _Conv(c, c, 3, 1)
        self.cv2 = _Conv(c, c, 3, 1)
        self.add = shortcut

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class _C2fN1(nn.Module):
    """C2f with n=1 bottleneck — the most common variant in YOLOv8n.

        y0, y1 = cv1(x).chunk(2, dim=1)
        y2 = m0(y1)
        return cv2(cat([y0, y1, y2], dim=1))
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = _Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = _Conv(3 * self.c, c2, 1, 1)
        self.m0 = _Bottleneck(self.c, shortcut)

    def forward(self, x):
        t = self.cv1(x)
        y0, y1 = t.chunk(2, 1)
        y2 = self.m0(y1)
        return self.cv2(torch.cat([y0, y1, y2], 1))


class _C2fN2(nn.Module):
    """C2f with n=2 bottlenecks — used in backbone layers 4 and 6 only."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = _Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = _Conv(4 * self.c, c2, 1, 1)
        self.m0 = _Bottleneck(self.c, shortcut)
        self.m1 = _Bottleneck(self.c, shortcut)

    def forward(self, x):
        t = self.cv1(x)
        y0, y1 = t.chunk(2, 1)
        y2 = self.m0(y1)
        y3 = self.m1(y2)
        return self.cv2(torch.cat([y0, y1, y2, y3], 1))


class _SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast: 3 chained 5×5 max-pools concat'd
    with the input. The maxpool has stride=1, padding=k//2 so spatial
    dims are preserved across all 4 branches (input + 3 pooled)."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = _Conv(c1, c_, 1, 1)
        self.cv2 = _Conv(4 * c_, c2, 1, 1)
        self.m1 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.m2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.m3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m1(x)
        y2 = self.m2(y1)
        y3 = self.m3(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class _DetectHead(nn.Module):
    """Simplified Detect head: per-scale (cv2 box, cv3 cls) towers,
    output is cat([box, cls], dim=1) per scale. Returns the 3 raw
    detection feature maps as a tuple — no DFL decode, no NMS, no
    end2end fork. Those live in Python post-processing where they
    belong.

    Channel layout per scale: [reg_max*4 (box) | nc (cls)]. For the
    stock YOLOv8n with reg_max=16 nc=80 that's [64 | 80] = 144 ch.
    """

    def __init__(self, ch: tuple[int, int, int] = (64, 128, 256),
                 nc: int = 80, reg_max: int = 16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(nc, 100))
        for s, in_c in enumerate(ch):
            setattr(self, f"cv2_{s}_0", _Conv(in_c, c2, 3, 1))
            setattr(self, f"cv2_{s}_1", _Conv(c2, c2, 3, 1))
            setattr(self, f"cv2_{s}_2", nn.Conv2d(c2, 4 * reg_max, 1))
            setattr(self, f"cv3_{s}_0", _Conv(in_c, c3, 3, 1))
            setattr(self, f"cv3_{s}_1", _Conv(c3, c3, 3, 1))
            setattr(self, f"cv3_{s}_2", nn.Conv2d(c3, nc, 1))

    def forward(self, x_p3, x_p4, x_p5):
        b0 = self.cv2_0_2(self.cv2_0_1(self.cv2_0_0(x_p3)))
        c0 = self.cv3_0_2(self.cv3_0_1(self.cv3_0_0(x_p3)))
        out0 = torch.cat([b0, c0], 1)
        b1 = self.cv2_1_2(self.cv2_1_1(self.cv2_1_0(x_p4)))
        c1 = self.cv3_1_2(self.cv3_1_1(self.cv3_1_0(x_p4)))
        out1 = torch.cat([b1, c1], 1)
        b2 = self.cv2_2_2(self.cv2_2_1(self.cv2_2_0(x_p5)))
        c2v = self.cv3_2_2(self.cv3_2_1(self.cv3_2_0(x_p5)))
        out2 = torch.cat([b2, c2v], 1)
        return out0, out1, out2


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class YOLOv8Nano(nn.Module):
    """YOLOv8-nano — flat-forward variant of ultralytics yolov8n.

    Layer numbering matches the upstream YAML:
        0..9   backbone (Conv/C2f/SPPF chain)
        10..21 PAN-FPN neck (upsamples, concats, C2fs)
        22     detection head (3 scales)

    Layers 11/14/17/20 are pure cat ops (no parameters), so they
    aren't allocated as submodules — they appear as torch.cat calls
    in forward(). Layer 22's outputs are 3 tensors covering the P3,
    P4, P5 detection scales.
    """

    def __init__(self, nc: int = 80):
        super().__init__()
        # backbone
        self.l0 = _Conv(3, 16, 3, 2)
        self.l1 = _Conv(16, 32, 3, 2)
        self.l2 = _C2fN1(32, 32, shortcut=True)
        self.l3 = _Conv(32, 64, 3, 2)
        self.l4 = _C2fN2(64, 64, shortcut=True)
        self.l5 = _Conv(64, 128, 3, 2)
        self.l6 = _C2fN2(128, 128, shortcut=True)
        self.l7 = _Conv(128, 256, 3, 2)
        self.l8 = _C2fN1(256, 256, shortcut=True)
        self.l9 = _SPPF(256, 256, 5)
        # neck
        self.l10 = nn.Upsample(scale_factor=2, mode="nearest")
        self.l12 = _C2fN1(256 + 128, 128, shortcut=False)
        self.l13 = nn.Upsample(scale_factor=2, mode="nearest")
        self.l15 = _C2fN1(128 + 64, 64, shortcut=False)
        self.l16 = _Conv(64, 64, 3, 2)
        self.l18 = _C2fN1(64 + 128, 128, shortcut=False)
        self.l19 = _Conv(128, 128, 3, 2)
        self.l21 = _C2fN1(128 + 256, 256, shortcut=False)
        # head
        self.detect = _DetectHead(ch=(64, 128, 256), nc=nc)

    def forward(self, x):
        # backbone
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x4 = self.l4(x)        # save: P3 source
        x = self.l5(x4)
        x6 = self.l6(x)        # save: P4 source
        x = self.l7(x6)
        x = self.l8(x)
        x9 = self.l9(x)        # save: P5 source
        # neck
        u10 = self.l10(x9)
        cat11 = torch.cat([u10, x6], 1)
        x12 = self.l12(cat11)
        u13 = self.l13(x12)
        cat14 = torch.cat([u13, x4], 1)
        x15 = self.l15(cat14)
        x16 = self.l16(x15)
        cat17 = torch.cat([x16, x12], 1)
        x18 = self.l18(cat17)
        x19 = self.l19(x18)
        cat20 = torch.cat([x19, x9], 1)
        x21 = self.l21(cat20)
        # detect head — 3 raw detection scale outputs
        return self.detect(x15, x18, x21)


# ---------------------------------------------------------------------------
# Weight loading from ultralytics yolov8n.pt
# ---------------------------------------------------------------------------


# Layer-index → (kind, our-attr-name).  None means "no params".
_LAYER_MAP = {
    0:  ("conv", "l0"),  1:  ("conv", "l1"),  2:  ("c2f1", "l2"),
    3:  ("conv", "l3"),  4:  ("c2f2", "l4"),  5:  ("conv", "l5"),
    6:  ("c2f2", "l6"),  7:  ("conv", "l7"),  8:  ("c2f1", "l8"),
    9:  ("sppf", "l9"),
    10: (None,   None),  11: (None,    None),
    12: ("c2f1", "l12"), 13: (None,   None),  14: (None,   None),
    15: ("c2f1", "l15"), 16: ("conv", "l16"), 17: (None,   None),
    18: ("c2f1", "l18"), 19: ("conv", "l19"), 20: (None,   None),
    21: ("c2f1", "l21"),
    22: ("detect", "detect"),
}


def _ultra_to_local_key(ultra_key: str) -> Optional[str]:
    """Translate an ultralytics state_dict key like
        model.4.m.0.cv2.bn.weight   →  l4.m0.cv2.bn.weight
        model.22.cv2.0.0.conv.weight →  detect.cv2_0_0.conv.weight
    Returns None for keys that don't belong to a parameterized layer
    (e.g. layer 10 nn.Upsample, layer 11 Concat, the DFL inside Detect)."""
    m = re.match(r"^model\.(\d+)\.(.*)$", ultra_key)
    if not m:
        return None
    layer_idx = int(m.group(1))
    rest = m.group(2)
    kind_attr = _LAYER_MAP.get(layer_idx, (None, None))
    kind, attr = kind_attr
    if kind is None:
        return None

    if kind == "conv":
        # rest is e.g. "conv.weight" or "bn.running_mean"
        return f"{attr}.{rest}"

    if kind in ("c2f1", "c2f2"):
        # rest paths:
        #   cv1.conv.weight       →  cv1.conv.weight   (direct)
        #   m.0.cv1.conv.weight   →  m0.cv1.conv.weight
        m2 = re.match(r"^m\.(\d+)\.(.*)$", rest)
        if m2:
            return f"{attr}.m{m2.group(1)}.{m2.group(2)}"
        return f"{attr}.{rest}"

    if kind == "sppf":
        # rest: cv1.conv.weight, cv2.bn.bias, etc. m.* has no params.
        if rest.startswith("m."):
            return None
        return f"{attr}.{rest}"

    if kind == "detect":
        # rest: cv2.<scale>.<idx>.<...>  or cv3.<scale>.<idx>.<...>
        #   cv2.0.0.conv.weight   →  cv2_0_0.conv.weight
        #   cv2.0.2.weight        →  cv2_0_2.weight
        m2 = re.match(r"^(cv[23])\.(\d+)\.(\d+)\.(.*)$", rest)
        if m2:
            br, scale, idx, tail = m2.groups()
            return f"{attr}.{br}_{scale}_{idx}.{tail}"
        # dfl.* and other sub-modules: ignore
        return None

    return None


def _load_ultralytics_weights(model: YOLOv8Nano) -> int:
    """Stream weights from yolov8n.pt into model. Returns count copied."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(
            "ultralytics not installed. `pip install ultralytics` and retry, "
            "or set AGENTS_YOLOV8N_PRETRAINED=0 to use random init."
        ) from e
    yolo = YOLO("yolov8n.pt")
    src_state = yolo.model.state_dict()
    dst_state = model.state_dict()
    n_copied = 0
    for k_src, v_src in src_state.items():
        k_dst = _ultra_to_local_key(k_src)
        if k_dst is None:
            continue
        if k_dst not in dst_state:
            continue
        if dst_state[k_dst].shape != v_src.shape:
            raise RuntimeError(
                f"shape mismatch loading {k_src} → {k_dst}: "
                f"src {tuple(v_src.shape)} vs dst {tuple(dst_state[k_dst].shape)}"
            )
        dst_state[k_dst].copy_(v_src)
        n_copied += 1
    model.load_state_dict(dst_state)
    return n_copied


# ---------------------------------------------------------------------------
# Module entry points (matching the agents/models/<x>.py interface)
# ---------------------------------------------------------------------------


def _cfg() -> tuple[int, int, bool]:
    img = int(os.environ.get("AGENTS_YOLOV8N_INPUT", 160))
    if img % 32 != 0:
        raise SystemExit(
            f"AGENTS_YOLOV8N_INPUT={img} must be a multiple of 32 "
            f"(YOLOv8 stride-32 head requires it)."
        )
    nc = int(os.environ.get("AGENTS_YOLOV8N_NC", 80))
    pretrained = os.environ.get("AGENTS_YOLOV8N_PRETRAINED", "1") == "1"
    return img, nc, pretrained


def get_model(seed: int = 0):
    img, nc, pretrained = _cfg()
    torch.manual_seed(seed)
    m = YOLOv8Nano(nc=nc)
    if pretrained:
        if nc != 80:
            # Backbone+neck weights still load; the cv3 head's last conv has
            # nc-dependent shape and would shape-mismatch. Refuse to silently
            # skip — force the user to acknowledge.
            raise SystemExit(
                f"AGENTS_YOLOV8N_NC={nc} ≠ 80 with pretrained weights: cv3 "
                f"head shapes don't match. Set AGENTS_YOLOV8N_PRETRAINED=0 "
                f"or fine-tune from a custom checkpoint (out of scope here)."
            )
        n = _load_ultralytics_weights(m)
        print(f"yolov8_nano: loaded {n} pretrained tensors from yolov8n.pt")
    m.eval()
    return m


def get_sample_input(seed: int = 1) -> torch.Tensor:
    """Synthetic NCHW frame at the configured resolution. The downstream
    pipeline uses the SAME random seed to regenerate this for the
    PyTorch golden, so the on-spike compare is apples-to-apples."""
    img, _, _ = _cfg()
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 3, img, img, generator=g)

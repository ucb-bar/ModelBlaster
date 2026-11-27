"""YOLOv8-x for the modelblaster flow (the forest stack's sized-down detector at 320 px).

Same flat, fx-traceable layout as models/yolov8_nano.py (whose building blocks and ultralytics
weight-key mapping it reuses), at YOLOv8x scale: width 1.25 with max_channels 512*1.25 = 640, depth 1.0
(C2f n = 3, 6, 6, 3 in the backbone and 3 in the neck). Raw 3-scale head outputs, no DFL / NMS.

Env knobs (same names as the nano wrapper, so calibration/preprocess settings carry over):
  MODELBLASTER_YOLOV8N_INPUT   default 320 here
  MODELBLASTER_YOLOV8X_WEIGHTS default yolov8x.pt (ultralytics COCO-80 release)
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from modelblaster.models import yolov8_nano as _n
from modelblaster.models.yolov8_nano import _Conv, _Bottleneck, _SPPF, _DetectHead

WEIGHTS = os.environ.get("MODELBLASTER_YOLOV8X_WEIGHTS", "yolov8x.pt")
os.environ.setdefault("MODELBLASTER_YOLOV8N_INPUT", "320")


class _C2f(nn.Module):
    """C2f with n bottlenecks, statically unrolled (m0..m{n-1})."""

    def __init__(self, c1: int, c2: int, n: int, shortcut: bool = True):
        super().__init__()
        self.c = c2 // 2
        self.n = n
        self.cv1 = _Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = _Conv((2 + n) * self.c, c2, 1, 1)
        for i in range(n):
            setattr(self, f"m{i}", _Bottleneck(self.c, shortcut))

    def forward(self, x):
        t = self.cv1(x)
        y0, y1 = t.chunk(2, 1)
        ys = [y0, y1]
        y = y1
        for i in range(self.n):
            y = getattr(self, f"m{i}")(y)
            ys.append(y)
        # the int8 extractor lowers 2/3/4-input cats only: nest them (same values in fp32)
        while len(ys) > 4:
            ys = [torch.cat(ys[:4], 1)] + ys[4:]
        return self.cv2(torch.cat(ys, 1))


class YOLOv8X(nn.Module):
    def __init__(self, nc: int = 80):
        super().__init__()
        self.l0 = _Conv(3, 80, 3, 2)
        self.l1 = _Conv(80, 160, 3, 2)
        self.l2 = _C2f(160, 160, 3, True)
        self.l3 = _Conv(160, 320, 3, 2)
        self.l4 = _C2f(320, 320, 6, True)
        self.l5 = _Conv(320, 640, 3, 2)
        self.l6 = _C2f(640, 640, 6, True)
        self.l7 = _Conv(640, 640, 3, 2)
        self.l8 = _C2f(640, 640, 3, True)
        self.l9 = _SPPF(640, 640, 5)
        self.l10 = nn.Upsample(scale_factor=2, mode="nearest")
        self.l12 = _C2f(640 + 640, 640, 3, False)
        self.l13 = nn.Upsample(scale_factor=2, mode="nearest")
        self.l15 = _C2f(640 + 320, 320, 3, False)
        self.l16 = _Conv(320, 320, 3, 2)
        self.l18 = _C2f(320 + 640, 640, 3, False)
        self.l19 = _Conv(640, 640, 3, 2)
        self.l21 = _C2f(640 + 640, 640, 3, False)
        self.detect = _DetectHead(ch=(320, 640, 640), nc=nc)

    forward = _n.YOLOv8Nano.forward


def _key(k: str):
    """ultralytics key -> local key; C2f layers use m<i> like the nano mapping (it covers any n)."""
    return _n._ultra_to_local_key(k)


def get_model(seed: int = 0):
    torch.manual_seed(seed)
    m = YOLOv8X(nc=80)
    from ultralytics import YOLO  # noqa: PLC0415
    src = YOLO(WEIGHTS).model.state_dict()
    dst = m.state_dict()
    n = 0
    for k, v in src.items():
        kd = _key(k)
        if kd is None or kd not in dst:
            continue
        if dst[kd].shape != v.shape:
            raise RuntimeError(f"shape mismatch {k} -> {kd}: {tuple(v.shape)} vs {tuple(dst[kd].shape)}")
        dst[kd].copy_(v)
        n += 1
    missing = [k for k in dst if not k.endswith("num_batches_tracked") and k not in
               {_key(s) for s in src}]
    if missing:
        raise RuntimeError(f"yolov8_x: {len(missing)} tensors not loaded, e.g. {missing[:5]}")
    m.load_state_dict(dst)
    print(f"yolov8_x: loaded {n} pretrained tensors from {WEIGHTS}")
    m.eval()
    return m


get_sample_input = _n.get_sample_input
get_calibration_spec = _n.get_calibration_spec
get_calibration_samples = _n.get_calibration_samples

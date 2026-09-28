"""SignNet GTSRB street-sign classifier wrapper for the modelblaster flow.

Sources the canonical PyTorch class from the vendored gtsrb_arch.py (see that
module's docstring) and loads the trained GRAYSCALE checkpoint by default. The
deployed graph is SignNetSoftmax(SignNet(...)) — base logits + a softmax head —
so the int8 flow exercises softmax_s8, matching the standalone eval_int8.py
that validated float 99.0% / int8 98.97% (STOP recall 1.0) with the faithful
fold_conv_bn=False extractor.

Deploy config (the io.npz golden anchor + IR shapes):

    in_ch = 1   input_size = 48   n_classes = 43   ->  [1, 1, 48, 48]

Preprocessing MIRRORS the standalone data.py byte-for-byte so float, calib and
deploy paths see identical inputs: Resize((48, 48)), Grayscale(1), ToTensor()
(raw [0, 1], NO mean/std normalization — matches the HM01B0 deploy path of raw
pixels / 255).

Override the checkpoint path via MODELBLASTER_GTSRB_CKPT, and the GTSRB dataset
root (for calibration / the golden anchor) via MODELBLASTER_GTSRB_DATA.
"""

from __future__ import annotations

import os

import torch

from . import gtsrb_arch

# Committed alongside this module (mirrors models/dronet.py) — previously an
# absolute path into one user's scratch dir, which doesn't exist on any other
# checkout.
_DEFAULT_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints", "gtsrb", "best.pt",
)

# GTSRB dataset root for calibration / golden-anchor loading. Defaults to the
# standalone gtsrb_signclf data dir (the same ROOT data.py used); override with
# MODELBLASTER_GTSRB_DATA to point at any torchvision.datasets.GTSRB layout.
_DEFAULT_DATA = "/scratch2/dima/misc_sw/gtsrb_signclf/data"
_DATA_ROOT = os.environ.get("MODELBLASTER_GTSRB_DATA", _DEFAULT_DATA)

# Fixed grayscale deploy config — the trained checkpoint is 1-channel 48x48.
_IN_CH = 1
_INPUT = 48
_N_CLASSES = 43

# GTSRB class ids of the demo signs (see the standalone data.py).
STOP = 14
YIELD = 13


def get_model(seed: int = 0):
    """SignNetSoftmax with the trained grayscale weights loaded, in eval()."""
    torch.manual_seed(seed)
    base = gtsrb_arch.SignNet(
        in_ch=_IN_CH, n_classes=_N_CLASSES, input_size=_INPUT)
    ckpt_path = os.environ.get("MODELBLASTER_GTSRB_CKPT", _DEFAULT_CKPT)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"GTSRB checkpoint not found at {ckpt_path}. "
            f"Set MODELBLASTER_GTSRB_CKPT to override."
        )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # The standalone trainer saves {"state_dict": ...}; tolerate a bare or
    # "model_state_dict"-wrapped dict too.
    if isinstance(ck, dict) and "state_dict" in ck:
        sd = ck["state_dict"]
    elif isinstance(ck, dict) and "model_state_dict" in ck:
        sd = ck["model_state_dict"]
    else:
        sd = ck
    base.load_state_dict(sd, strict=True)
    base.eval()
    m = gtsrb_arch.SignNetSoftmax(base)
    m.eval()
    return m


def _get_transforms():
    """Grayscale 48x48 transforms — byte-identical to data.py (no aug)."""
    from torchvision import transforms  # noqa: PLC0415
    return transforms.Compose([
        transforms.Resize((_INPUT, _INPUT)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])


def _train_tensors(n: int) -> list[torch.Tensor]:
    """`n` real GTSRB *train* images, preprocessed like data.py, each as a
    [1, 1, 48, 48] tensor. Indices are evenly spaced (np.linspace) over the
    train split — the same selection eval_int8.py used, so index 0 (== the
    golden anchor) is train image 0."""
    import numpy as np  # noqa: PLC0415
    from torchvision.datasets import GTSRB  # noqa: PLC0415

    ds = GTSRB(root=_DATA_ROOT, split="train", download=False,
               transform=_get_transforms())
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    return [ds[int(i)][0].unsqueeze(0) for i in idx]


def get_sample_input(seed: int = 1) -> torch.Tensor:
    """One real GTSRB train frame, NCHW grayscale [1, 1, 48, 48] in [0, 1].

    Real-image input makes a meaningful io.npz golden even at
    --num-calibration 1. Falls back to a deterministic torch.randn only if the
    GTSRB dataset isn't available (with a warning); with --num-calibration > 1
    the golden anchor is get_calibration_samples()[0] anyway."""
    try:
        return _train_tensors(1)[0]
    except Exception as e:  # noqa: BLE001
        print(f"[gtsrb.get_sample_input] WARN: couldn't load GTSRB from "
              f"{_DATA_ROOT} ({e}); falling back to torch.randn. Set "
              f"MODELBLASTER_GTSRB_DATA to a GTSRB root to fix.")
        g = torch.Generator().manual_seed(seed)
        return torch.randn(1, _IN_CH, _INPUT, _INPUT, generator=g)


def get_calibration_samples(n: int = 32) -> list[torch.Tensor]:
    """`n` real GTSRB train images for int8 PTQ activation calibration,
    preprocessed IDENTICALLY to data.py (grayscale, 48x48, ToTensor [0,1]).

    Each sample is a [1, 1, 48, 48] tensor; the first becomes the io.npz
    golden anchor. More samples widen the per-tensor activation ranges so the
    int8 scales reflect the true distribution (not a single frame)."""
    return _train_tensors(n)

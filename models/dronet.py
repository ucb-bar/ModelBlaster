"""DroNet model wrapper for the modelblaster flow.

Sources the canonical PyTorch class from the vendored dronet_arch.py (see
that module's docstring) and loads the latest trained checkpoint by
default. The trained config is:

    img_dims = (112, 112)   img_channels = 3   small = True
    -> linear_in = 2048 (after the conv/pool stack).

Both 128×128×1 (the original demo) and 112×112×3 (the trained config) yield
linear_in=2048 with small=True; we follow the trained config so the loaded
weights match.

Override the checkpoint path via MODELBLASTER_DRONET_CKPT. For a scaled or
grayscale geometry (see below) the checkpoint is only read when that variable
names one, and it must then match the geometry exactly.

MODELBLASTER_DRONET_CALIB=<file.npy> supplies real int8 PTQ calibration frames
(float32 NCHW, N x C x H x W, matching the geometry): get_sample_input() returns
its first frame and get_calibration_samples(n) evenly spaced frames from it,
which extract_graph uses with --num-calibration n.
"""

from __future__ import annotations

import os

import torch

from . import dronet_arch

# Committed alongside this module (see models/mlp_control.py for the same
# fix applied there) -- previously an absolute path into one user's home
# directory, which doesn't exist on any other checkout.
_DEFAULT_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints", "dronet", "best.pt",
)

# Trained config: 3-channel input at 112x112.
#
# MODELBLASTER_DRONET_INPUT rescales the input for SCALED VARIANTS used in
# scheduling / performance studies. The trained checkpoint only fits the
# default 112, so a rescaled model is left at its seeded random init: the
# golden is generated from those same weights, so bit-exactness still checks
# and the compute SHAPE is what the variant exists to vary. Its predictions
# are meaningless -- never quote a scaled variant as an accuracy result.
# MODELBLASTER_DRONET_CHANNELS selects the input channel count for GRAYSCALE /
# performance variants (default 3 = the trained RGB config, unchanged). A
# non-3 channel count cannot match the trained checkpoint's conv0 shape, so it
# is handled exactly like a rescaled geometry below: seeded random init, golden
# generated from the same weights, so correctness still checks and the compute
# SHAPE (grayscale = 1ch conv0) is what the variant exists to vary. Never quote
# a grayscale/scaled variant as an accuracy result.
_IMG_CHANNELS = int(os.environ.get("MODELBLASTER_DRONET_CHANNELS", "3"))
_IMG_H = int(os.environ.get("MODELBLASTER_DRONET_INPUT", "112"))
_IMG_W = _IMG_H
_SCALED = (_IMG_H != 112) or (_IMG_CHANNELS != 3)
_CALIB = os.environ.get("MODELBLASTER_DRONET_CALIB")


def _load_matched_checkpoint(m, ckpt_path: str) -> None:
    """Load a checkpoint trained for THIS geometry into m, or raise.

    Used for scaled / grayscale variants, which are otherwise left at seeded
    random init. A checkpoint was explicitly asked for, so every key and every
    shape must match: no partial load, and never a silent fall back to random
    weights (the committed grayscale model once shipped random weights that
    way, see the note above).
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"MODELBLASTER_DRONET_CKPT={ckpt_path} does not exist")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    want = m.state_dict()
    missing = sorted(set(want) - set(sd))
    unexpected = sorted(set(sd) - set(want))
    shapes = [f"{k}: checkpoint {tuple(sd[k].shape)} != model {tuple(v.shape)}"
              for k, v in want.items()
              if k in sd and tuple(sd[k].shape) != tuple(v.shape)]
    if missing or unexpected or shapes:
        raise RuntimeError(
            f"DroNet checkpoint {ckpt_path} does not match the "
            f"{_IMG_CHANNELS}x{_IMG_H}x{_IMG_W} geometry: "
            f"missing={missing[:8]} unexpected={unexpected[:8]} "
            f"shape={shapes[:8]}")
    m.load_state_dict(sd, strict=True)
    print(f"[dronet] loaded {_IMG_CHANNELS}x{_IMG_H}x{_IMG_W} checkpoint "
          f"{ckpt_path}", flush=True)


def get_model(seed: int = 0):
    torch.manual_seed(seed)
    m = dronet_arch.DronetTorch(
        img_dims=(_IMG_H, _IMG_W),
        img_channels=_IMG_CHANNELS,
        output_dim=1,
        small=True,
    )
    if _SCALED:
        # Rescaled geometry cannot match the default checkpoint's shapes; only
        # an explicitly named, geometry-matched checkpoint is loaded.
        ckpt_path = os.environ.get("MODELBLASTER_DRONET_CKPT")
        if ckpt_path:
            _load_matched_checkpoint(m, ckpt_path)
        m.eval()
        return m
    ckpt_path = os.environ.get("MODELBLASTER_DRONET_CKPT", _DEFAULT_CKPT)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"DroNet checkpoint not found at {ckpt_path}. "
            f"Set MODELBLASTER_DRONET_CKPT to override."
        )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in DroNet checkpoint: {unexpected}")
    if missing:
        # The trained checkpoint only has the steering head trained; linear2
        # (collision) is at random init in the checkpoint AND in our freshly
        # built model. They might differ — that's fine for steering inference,
        # but warn if other keys are missing.
        non_linear2 = [k for k in missing if not k.startswith("linear2.")]
        if non_linear2:
            raise RuntimeError(
                f"DroNet checkpoint missing weights: {non_linear2[:8]}"
            )
    m.eval()
    return m


def _calib_frames():
    import numpy as np  # noqa: PLC0415
    a = np.load(_CALIB)
    want = (_IMG_CHANNELS, _IMG_H, _IMG_W)
    if a.ndim != 4 or tuple(a.shape[1:]) != want or a.shape[0] < 1:
        raise RuntimeError(
            f"MODELBLASTER_DRONET_CALIB={_CALIB}: shape {a.shape}, "
            f"want N x {want[0]} x {want[1]} x {want[2]}")
    # Normalised frames only: raw uint8 pixels (or 0..255 floats) would
    # calibrate every activation range ~127x too wide without any error.
    if not np.issubdtype(a.dtype, np.floating):
        raise RuntimeError(
            f"MODELBLASTER_DRONET_CALIB={_CALIB}: dtype {a.dtype}, want float "
            f"frames x = clamp(p - 128, -127, 127) / 127")
    if not np.isfinite(a).all():
        raise RuntimeError(f"MODELBLASTER_DRONET_CALIB={_CALIB}: non-finite values")
    lo, hi = float(a.min()), float(a.max())
    if lo < -1.0 or hi > 1.0:
        raise RuntimeError(
            f"MODELBLASTER_DRONET_CALIB={_CALIB}: values span [{lo:g}, {hi:g}], "
            f"want [-1, 1] (x = clamp(p - 128, -127, 127) / 127)")
    return a.astype(np.float32, copy=False)


def get_sample_input(seed: int = 1) -> torch.Tensor:
    """Synthetic NCHW frame matching the trained input shape, or the first
    MODELBLASTER_DRONET_CALIB frame when that is set."""
    if _CALIB:
        return torch.from_numpy(_calib_frames()[:1].copy())
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, _IMG_CHANNELS, _IMG_H, _IMG_W, generator=g)


if _CALIB:
    def get_calibration_samples(n: int = 32) -> list:
        """`n` evenly spaced MODELBLASTER_DRONET_CALIB frames, each a
        [1, C, H, W] tensor; the first (frame 0) is the io.npz golden anchor.
        Defined only when the variable is set, so without it
        --num-calibration keeps its old single-sample fallback."""
        import numpy as np  # noqa: PLC0415
        a = _calib_frames()
        if n > a.shape[0]:
            raise RuntimeError(
                f"--num-calibration {n} but {_CALIB} has {a.shape[0]} frames")
        idx = np.linspace(0, a.shape[0] - 1, n).round().astype(int)
        return [torch.from_numpy(a[i:i + 1].copy()) for i in idx]

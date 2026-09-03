"""DroNet model wrapper for the modelblaster flow.

Sources the canonical PyTorch class from the vendored dronet_arch.py (see
that module's docstring) and loads the latest trained checkpoint by
default. The trained config is:

    img_dims = (112, 112)   img_channels = 3   small = True
    -> linear_in = 2048 (after the conv/pool stack).

Both 128×128×1 (the original demo) and 112×112×3 (the trained config) yield
linear_in=2048 with small=True; we follow the trained config so the loaded
weights match.

Override the checkpoint path via MODELBLASTER_DRONET_CKPT.
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
_IMG_CHANNELS = 3
_IMG_H = int(os.environ.get("MODELBLASTER_DRONET_INPUT", "112"))
_IMG_W = _IMG_H
_SCALED = (_IMG_H != 112)


def get_model(seed: int = 0):
    torch.manual_seed(seed)
    m = dronet_arch.DronetTorch(
        img_dims=(_IMG_H, _IMG_W),
        img_channels=_IMG_CHANNELS,
        output_dim=1,
        small=True,
    )
    if _SCALED:
        # Rescaled geometry cannot match the trained checkpoint's shapes.
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


def get_sample_input(seed: int = 1) -> torch.Tensor:
    """Synthetic NCHW frame matching the trained input shape."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, _IMG_CHANNELS, _IMG_H, _IMG_W, generator=g)

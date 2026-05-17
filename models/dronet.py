"""DroNet model wrapper for the modelblaster flow.

Sources the canonical PyTorch class from qnn_models/dronet.py and loads the
latest trained checkpoint by default. The trained config is:

    img_dims = (112, 112)   img_channels = 3   small = True
    -> linear_in = 2048 (after the conv/pool stack).

Both 128×128×1 (the original demo) and 112×112×3 (the trained config) yield
linear_in=2048 with small=True; we follow the trained config so the loaded
weights match.

Override the checkpoint path via MODELBLASTER_DRONET_CKPT.
"""

from __future__ import annotations

import importlib.util
import os

import torch


_DRONET_SRC = "/scratch2/dima/misc_sw/FreshScheduler/qnn_models/dronet.py"

# Latest trained DroNet checkpoint (best by val_mse from the 30-epoch run).
_DEFAULT_CKPT = (
    "/scratch2/dima/misc_sw/FreshScheduler/logs/dronet/2026-04-27_17-10-41/best.pt"
)

# Trained config: 3-channel input at 112x112.
_IMG_CHANNELS = 3
_IMG_H = 112
_IMG_W = 112


def _load_dronet_module():
    spec = importlib.util.spec_from_file_location("qnn_dronet", _DRONET_SRC)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load dronet from {_DRONET_SRC}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_model(seed: int = 0):
    torch.manual_seed(seed)
    mod = _load_dronet_module()
    m = mod.DronetTorch(
        img_dims=(_IMG_H, _IMG_W),
        img_channels=_IMG_CHANNELS,
        output_dim=1,
        small=True,
    )
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

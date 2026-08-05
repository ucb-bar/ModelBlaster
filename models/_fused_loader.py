"""Shared loader for the collaborator's FusedSensorNet (CNN variant).

Loads the deployable v12 CNN checkpoint into ``FusedSensorNet(vision_encoder=
"cnn", out_dim=2)``, folds the ``spectral_norm`` weight parametrizations, and
exposes the two single-input feed-forward sub-branches (vision, depth) as plain
conv/relu/linear wrappers that ModelBlaster can lower like DroNet.

The collaborator's files under /scratch/agustin are NOT modified; we only add
their directory to sys.path so ``fused_model`` and its ``ViTsubmodules`` import
resolve.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import torch
import torch.nn as nn

_VITFLY_MODELS = "/scratch/agustin/projects/DIMA/vitfly/models"
_FUSED_SRC = os.path.join(_VITFLY_MODELS, "fused_model.py")
_CKPT = (
    "/scratch/agustin/projects/DIMA/train_out/"
    "fused_bc_warehouse_v12_mixed_cnn/2026-08-03_19-51-49/best.pt"
)


def _load_fused_module():
    # fused_model.py does `from ViTsubmodules import ...` unconditionally, so the
    # collaborator's models dir must be importable.
    if _VITFLY_MODELS not in sys.path:
        sys.path.insert(0, _VITFLY_MODELS)
    spec = importlib.util.spec_from_file_location("fused_model", _FUSED_SRC)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load fused_model from {_FUSED_SRC}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fold_spectral_norm(module: nn.Module) -> None:
    """Bake old-style spectral_norm (weight_orig/weight_u/weight_v hooks) into a
    plain ``.weight`` on every submodule that carries it."""
    for m in module.modules():
        # Old-style spectral_norm registers a forward pre-hook and stores
        # weight_orig; remove_spectral_norm recomputes weight = weight_orig/sigma
        # and restores a plain Parameter named "weight".
        if hasattr(m, "weight_orig"):
            try:
                torch.nn.utils.remove_spectral_norm(m)
            except (ValueError, RuntimeError):
                # Fall back to the parametrize API if this is a new-style param.
                try:
                    torch.nn.utils.parametrize.remove_parametrizations(m, "weight")
                except Exception:
                    pass


def load_fused_cnn(seed: int = 0):
    """Return an eval-mode, spectral_norm-folded FusedSensorNet(cnn, out_dim=2)
    with the v12 trained weights loaded (strict=False; ViT keys are absent)."""
    torch.manual_seed(seed)
    fm = _load_fused_module()
    net = fm.FusedSensorNet(vision_encoder="cnn", out_dim=2)

    ckpt_path = os.environ.get("MODELBLASTER_FUSED_CKPT", _CKPT)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Fused checkpoint not found at {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    # strict=False: the "vit" encoder keys are absent (we built the cnn variant).
    net.load_state_dict(sd, strict=False)
    net.eval()
    _fold_spectral_norm(net)
    return net


class _VisionBranch(nn.Module):
    """front_grey (1,1,60,90) -> vision_cnn -> flatten(1536) -> vision_fc -> (1,512)."""

    def __init__(self, vision_cnn: nn.Module, vision_fc: nn.Module):
        super().__init__()
        self.vision_cnn = vision_cnn
        self.vision_fc = vision_fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.vision_cnn(x)
        f = torch.flatten(f, start_dim=1)
        return self.vision_fc(f)


class _DepthBranch(nn.Module):
    """tof_cross (1,4,8,8) -> depth_conv -> flatten(1024) -> depth_fc -> (1,64)."""

    def __init__(self, depth_conv: nn.Module, depth_fc: nn.Module):
        super().__init__()
        self.depth_conv = depth_conv
        self.depth_fc = depth_fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.depth_conv(x)
        f = torch.flatten(f, start_dim=1)
        return self.depth_fc(f)


def get_vision_branch(seed: int = 0) -> nn.Module:
    net = load_fused_cnn(seed)
    m = _VisionBranch(net.vision_cnn, net.vision_fc)
    m.eval()
    return m


def get_depth_branch(seed: int = 0) -> nn.Module:
    net = load_fused_cnn(seed)
    m = _DepthBranch(net.depth_conv, net.depth_fc)
    m.eval()
    return m

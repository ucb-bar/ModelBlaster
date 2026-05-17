"""Trained drone-control MLP policy.

Wraps the actor MLP from the rsl_rl PPO runner trained on the
crazyflie_steering_tracking task. Architecture comes from
SteeringTrackingPPORunnerCfg (sims/.../crazyflie/modelblaster/rsl_rl_ppo_cfg.py):

    obs_dim=16, action_dim=4, hidden_dims=[256, 128, 64], activation=elu

The checkpoint's actor_state_dict contains keys like `mlp.0.weight`, `mlp.2.weight`,
... which map to a torch.nn.Sequential(Linear, ELU, Linear, ELU, Linear, ELU, Linear)
exactly. We construct that module here and load the actor weights into it.

Inference is deterministic — just MLP forward (no sampling, no log_std).
"""

from __future__ import annotations

import os
import torch
from torch import nn


# Trained-model architecture (keep in sync with SteeringTrackingPPORunnerCfg).
_OBS_DIM = 16
_ACTION_DIM = 4
_HIDDEN_DIMS = [256, 128, 64]

# Latest trained MLP checkpoint at the time of writing. Updates pick up via
# MODELBLASTER_MLP_CONTROL_CKPT env var.
_DEFAULT_CKPT = (
    "/scratch2/dima/misc_sw/FreshScheduler/logs/rsl_rl/"
    "crazyflie_steering_tracking/2026-04-13_12-23-08/model_6998.pt"
)


def _build_mlp() -> nn.Sequential:
    """nn.Sequential with state-dict keys matching the trained checkpoint."""
    dims = [_OBS_DIM, *_HIDDEN_DIMS, _ACTION_DIM]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ELU(alpha=1.0))
    return nn.Sequential(*layers)


class MLPControl(nn.Module):
    """Inference-only wrapper. Forward is the deterministic actor."""

    def __init__(self):
        super().__init__()
        self.mlp = _build_mlp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def _load_actor_weights(model: MLPControl, ckpt_path: str) -> None:
    """Load `actor_state_dict` from an rsl_rl checkpoint into MLPControl.

    The checkpoint contains an extra `distribution.std_param` we ignore — it's
    only used for stochastic rollouts during training, not for inference.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "actor_state_dict" in ckpt:
        sd = ckpt["actor_state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt
    sd = {k: v for k, v in sd.items() if not k.startswith("distribution.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(
            f"unexpected keys in MLP control checkpoint: {unexpected}"
        )
    # All `mlp.{0,2,4,6}.{weight,bias}` should be present.
    expected_lins = {f"mlp.{i*2}.{p}"
                     for i in range(len(_HIDDEN_DIMS) + 1)
                     for p in ("weight", "bias")}
    if expected_lins - set(sd.keys()):
        raise RuntimeError(
            f"missing weights from checkpoint: {sorted(expected_lins - set(sd.keys()))}"
        )


def get_model(seed: int = 0) -> MLPControl:
    """Build + load the trained actor for inference."""
    torch.manual_seed(seed)
    m = MLPControl()
    ckpt_path = os.environ.get("MODELBLASTER_MLP_CONTROL_CKPT", _DEFAULT_CKPT)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"MLP control checkpoint not found at {ckpt_path}. "
            f"Set MODELBLASTER_MLP_CONTROL_CKPT to override."
        )
    _load_actor_weights(m, ckpt_path)
    m.eval()
    return m


def get_sample_input(seed: int = 1) -> torch.Tensor:
    """Synthetic 16-dim observation. Real obs is base_lin_vel(3) +
    base_ang_vel(3) + projected_gravity(3) + base_height(1) +
    target_command(2) + last_action(4) = 16."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, _OBS_DIM, generator=g)

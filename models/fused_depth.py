"""ModelBlaster wrapper: depth branch of FusedSensorNet (CNN variant).

depth_conv (2x Conv2d+ReLU) on tof_cross (1,4,8,8) -> flatten 16*8*8=1024 ->
depth_fc Linear 1024->64. Pure conv/relu/linear, single input = DroNet-shaped.
"""

from __future__ import annotations

import torch

from modelblaster.models._fused_loader import get_depth_branch

_C, _H, _W = 4, 8, 8


def get_model(seed: int = 0):
    return get_depth_branch(seed)


def get_sample_input(seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, _C, _H, _W, generator=g)

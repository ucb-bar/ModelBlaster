"""ModelBlaster wrapper: vision branch of FusedSensorNet (CNN variant).

vision_cnn (4x Conv2d+ReLU) on front_grey (1,1,60,90) -> flatten 64*4*6=1536 ->
vision_fc Linear 1536->512. Pure conv/relu/linear, single input = DroNet-shaped.
"""

from __future__ import annotations

import torch

from modelblaster.models._fused_loader import get_vision_branch

_H, _W = 60, 90


def get_model(seed: int = 0):
    return get_vision_branch(seed)


def get_sample_input(seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 1, _H, _W, generator=g)

"""Tiny MLP for the first end-to-end example."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features: int = 16, hidden: int = 32, out_features: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def get_model(seed: int = 0) -> MLP:
    torch.manual_seed(seed)
    m = MLP()
    m.eval()
    return m


def get_sample_input(seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 16, generator=g)

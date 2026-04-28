"""LeNet-5 style CNN for the conv2d/maxpool2d stepping stone.

Same shapes as the LeNet variant in samples/executorch/model/gen_pte.py:
1x28x28 -> conv1(1->6, k=5) -> relu -> pool(2,2) -> conv2(6->16, k=5)
   -> relu -> pool(2,2) -> flatten -> fc(256->120) -> relu -> fc(120->84)
   -> relu -> fc(84->10)
"""

from __future__ import annotations

import torch
from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def get_model(seed: int = 0) -> LeNet:
    torch.manual_seed(seed)
    m = LeNet()
    m.eval()
    return m


def get_sample_input(seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 1, 28, 28, generator=g)

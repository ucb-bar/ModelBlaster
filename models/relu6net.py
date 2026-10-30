"""Tiny conv/bn/ReLU6 net — the stepping stone for int8 ReLU6 support.

Exercises BOTH the nn.ReLU6 module and the torch.nn.functional.relu6
functional forms (the two extractor branches), plus plain conv+bn+maxpool+
linear so the whole int8 pipeline runs end to end. The batchnorm affine is
initialised to push activations above 6.0 so the ReLU6 upper clamp genuinely
engages (qmax < 127) rather than degenerating to a plain relu.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ReLU6Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.act1 = nn.ReLU6()                      # module form
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 4 * 4, 10)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = F.relu6(self.bn2(self.conv2(x)))        # functional form
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


def get_model(seed: int = 0) -> ReLU6Net:
    torch.manual_seed(seed)
    m = ReLU6Net()
    # Fold the batchnorms into eval mode with running stats that make the
    # post-bn activations span well beyond 6, so ReLU6 clamps.
    with torch.no_grad():
        for bn in (m.bn1, m.bn2):
            bn.weight.fill_(4.0)
            bn.bias.fill_(2.0)
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0)
    m.eval()
    return m


def get_sample_input(seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 3, 16, 16, generator=g)

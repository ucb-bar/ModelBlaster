"""A transformer-style block exercising the norm kernels SmolVLA needs.

SmolVLA's uncovered compute nodes cluster into families; this covers two of them
end to end so `layernorm_s8` and `rmsnorm_s8` are verified on hardware rather
than added as untested kernels:

  layer_norm      75 nodes  -> nn.LayerNorm
  pow + rsqrt    173 nodes  -> nn.RMSNorm (that pair IS RMSNorm written out)

Shapes are small so the golden comparison is quick; the ops are shape-agnostic.
"""

import torch
import torch.nn as nn


class NormBlock(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d)
        self.gelu = nn.GELU()
        self.rms = nn.RMSNorm(d) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d)
        self.fc2 = nn.Linear(d, 8)

    def forward(self, x):
        x = self.ln(x)
        x = self.gelu(self.fc1(x))
        x = self.rms(x)
        return self.fc2(x)


def get_model():
    # Fixed seed so get_model() is reproducible. Without it every call builds
    # different random weights, and anything comparing an extracted model
    # against a freshly constructed one is comparing two different networks --
    # which is exactly how int8 quality got measured at cosine 0.03 when the
    # real figure is 0.999.
    torch.manual_seed(0)
    m = NormBlock()
    m.eval()
    return m


def get_sample_input():
    return torch.randn(1, 64)

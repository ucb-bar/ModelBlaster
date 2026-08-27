"""A transformer attention block with RoPE, in the shape SmolVLA uses.

Covers the last two families in SmolVLA's op inventory end to end:

  scaled_dot_product_attention  36 nodes  -> matmul_s8 -> softmax_s8 -> matmul_s8
  sin + cos                     82 nodes  -> sin_s8 / cos_s8 (RoPE)

Attention is decomposed rather than given a fused kernel because matmul_s8
already carries `transpose_b` and `scale_div`, which is exactly QK^T/sqrt(d),
and softmax_s8 exists. Three existing kernels beat one new one duplicating them.

Single head and unmasked, matching what the extractor supports; the shapes are
small so the golden comparison is quick.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnBlock(nn.Module):
    def __init__(self, d=32, s=8):
        super().__init__()
        self.d, self.s = d, s
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        # The angle table is produced by a Linear rather than a registered
        # buffer: a buffer becomes a get_attr tensor-constant node, which the
        # int8 extractor does not yet materialise. That is a separate gap from
        # the ops under test here, and mixing the two would obscure which one
        # failed.
        self.ang = nn.Linear(d, d)

    def forward(self, x):
        x = self.norm(x)
        # RoPE: sin/cos of an angle table, applied multiplicatively.
        a = self.ang(x)
        rot = torch.sin(a) * torch.cos(a)
        q = self.q(x) * rot
        k = self.k(x) * rot
        v = self.v(x)
        a = F.scaled_dot_product_attention(q, k, v)
        return self.o(a)


def get_model():
    # Fixed seed so get_model() is reproducible. Without it every call builds
    # different random weights, and anything comparing an extracted model
    # against a freshly constructed one is comparing two different networks --
    # which is exactly how int8 quality got measured at cosine 0.03 when the
    # real figure is 0.999.
    torch.manual_seed(0)
    m = AttnBlock()
    m.eval()
    return m


def get_sample_input():
    return torch.randn(8, 32)   # [seq, feature], single head

"""A transformer MLP block, at the sequence length where the NPU earns its keep.

WHY THIS MODEL EXISTS, and it is a measurement rather than a preference.
`attn_block` was the transformer vehicle, and it cannot demonstrate
heterogeneous placement: its matmuls are M=8, its GEMM is 8.2% of the block,
and the K1's IME micro-tile is a hardware-forced 4 rows, so at M=8 the MAC unit
spends most of its width on padding. Measured on the board, IME is 0.85x RVV at
M=8 and 2.30x at M=128, crossing over at M=10.1. A scheduler offered both
implementations for attn_block therefore -- correctly -- never picks the NPU,
and the "heterogeneous schedule" has an empty accelerator lane.

The half of a transformer that DOES have large M is the MLP: the
position-wise feed-forward network runs one GEMM per token, so M is the
sequence length. That is this block, and it is not a synthetic stand-in --
`linear -> GELU -> linear` with a 4x inner expansion is exactly what sits after
attention in every transformer, ViNT's included.

SHAPES. seq=128, d_model=256, d_ff=1024 (the usual 4x). So the two GEMMs are
M=128,K=256,N=1024 and M=128,K=1024,N=256 -- both far above the M=10 crossover,
both large enough that the packing IME needs is amortized. d_model=256 keeps
the golden compare quick while leaving the shapes representative; the ops are
shape-agnostic.
"""

import torch
import torch.nn as nn

SEQ = 128
D_MODEL = 256
D_FF = 1024


class FFNBlock(nn.Module):
    def __init__(self, d_model=D_MODEL, d_ff=D_FF):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Pre-norm, as every modern transformer does it. The residual add is
        # deliberately included: it is a real dispatch and leaving it out would
        # make the GEMM fraction look larger than it is.
        h = self.ln(x)
        h = self.act(self.fc1(h))
        h = self.fc2(h)
        return x + h


def get_model():
    # Fixed seed so get_model() is reproducible. Without it every call builds
    # different random weights, and anything comparing an extracted model
    # against a freshly constructed one compares two different networks.
    torch.manual_seed(0)
    m = FFNBlock()
    m.eval()
    return m


def get_sample_input():
    # [seq, feature]. M is the sequence length, which is the whole point.
    return torch.randn(SEQ, D_MODEL)

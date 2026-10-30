"""Two-input int8 smoke model for the multi-input pipeline refactor.

Exercises N>1 typed input tensors end-to-end: two separately-shaped inputs each
go through their own Linear, are combined with an elementwise add (add_s8), then
ReLU + a head Linear. No cat/4D dependency, so it isolates the multi-input
plumbing (per-input placeholder, per-input scale/dtype, run_model N-arg ABI,
per-input io.npz golden) from other op quirks.
"""

import torch
import torch.nn as nn


class MultiToy(nn.Module):
    def __init__(self, da: int = 8, db: int = 4, h: int = 16, out: int = 4):
        super().__init__()
        self.fa = nn.Linear(da, h)
        self.fb = nn.Linear(db, h)
        self.head = nn.Linear(h, out)

    def forward(self, a, b):
        ha = self.fa(a)
        hb = self.fb(b)
        x = torch.relu(ha + hb)
        return self.head(x)


def get_model():
    torch.manual_seed(0)
    return MultiToy().eval()


def get_sample_input():
    # Placeholder order matches forward(a, b): a is (1,8), b is (1,4).
    torch.manual_seed(1)
    return (torch.randn(1, 8), torch.randn(1, 4))


def get_input_dtypes():
    # Both inputs quantized int8; the hook is here to exercise the multi-dtype
    # plumbing path (a real model could return e.g. ["i8", "f32"]).
    return ["i8", "i8"]

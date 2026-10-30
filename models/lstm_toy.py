"""Standalone int8 LSTM smoke model for the lstm_s8 kernel.

A multi-layer nn.LSTM over a seq-len-1 input, exercising the extractor's
per-layer lstm_s8 decomposition, the persistent h/c state buffers, and the
int8 LSTM kernel end-to-end. Output is the last layer's hidden sequence.
"""

import torch
import torch.nn as nn


class LstmToy(nn.Module):
    def __init__(self, inp: int = 8, hid: int = 4, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, layers)

    def forward(self, x):
        out, _ = self.lstm(x)   # x: (seq=1, batch=1, inp) -> out: (1,1,hid)
        return out


def get_model():
    torch.manual_seed(0)
    return LstmToy().eval()


def get_sample_input():
    torch.manual_seed(1)
    return torch.randn(1, 1, 8)   # (seq, batch, feat)

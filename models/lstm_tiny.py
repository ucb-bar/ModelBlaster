"""A minimal stateful LSTM, standing in for VitFly's recurrent core.

VitFly's LSTMNet uses LSTM(input_size=665, hidden_size=395, num_layers=2,
bias=False). This is the same construction at a size that keeps the golden
comparison quick, so the int8 LSTM path -- extractor, simulator, codegen and the
persistent hidden state -- is exercised end to end on hardware.

The point of interest is statefulness: `run_model` carries h and c across calls,
so invocation k depends on k-1. Every other model in the tree is a pure
function of its input.
"""

import torch
import torch.nn as nn


class LSTMTiny(nn.Module):
    def __init__(self, input_size=32, hidden_size=16, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bias=False)
        self.fc = nn.Linear(hidden_size, 4)

    def forward(self, x):
        y, _ = self.lstm(x)
        return self.fc(y)


def get_model():
    m = LSTMTiny()
    m.eval()
    return m


def get_sample_input():
    # [seq, batch, feature] -- one timestep, batch 1.
    return torch.randn(1, 1, 32)

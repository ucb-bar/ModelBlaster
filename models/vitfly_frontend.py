"""VitFly LSTMNet's convolutional front end, minus the LSTM.

The point of this model is to be exactly the part of
github.com/anish-bhattacharya/vitfly `LSTMNet` that ModelBlaster can already
lower: conv -> relu -> bn -> maxpool and conv -> relu -> bn -> avgpool, ending
at the flatten that would feed the LSTM. Shapes and layer parameters are taken
verbatim from the upstream definition.

It exists so `avgpool2d_s8` and `leaky_relu_s8` -- the two ops VitFly needs that
ModelBlaster lacked -- are exercised end to end on hardware, rather than being
added as untested kernels against a model that cannot yet be extracted. The
LSTM itself is a separate piece of work: it has no path in the int8 extractor,
and it needs hidden state carried across invocations, which the harness has no
concept of.
"""

import torch
import torch.nn as nn


class VitFlyFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        # verbatim from LSTMNet.__init__
        self.conv1 = nn.Conv2d(1, 4, 5, stride=3, padding=1)
        self.conv2 = nn.Conv2d(4, 10, 3, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(10)
        self.maxpool = nn.MaxPool2d(3, 1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        # LSTMNet applies leaky_relu after the LSTM's FC head; kept here so the
        # int8 path for it is covered by the same run.
        self.fc = nn.Linear(660, 64)  # flatten width for a 60x90 depth image
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.maxpool(x)
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.lrelu(self.fc(x))


def get_model():
    m = VitFlyFrontend()
    m.eval()
    return m


def get_sample_input():
    # LSTMNet consumes a single-channel depth image; 60x90 is the upstream size.
    return torch.randn(1, 1, 60, 90)

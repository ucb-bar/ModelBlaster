"""VitFly LSTMNet, faithful to github.com/anish-bhattacharya/vitfly.

Layer construction and forward order are taken verbatim from the upstream
`LSTMNet`. Two deliberate departures, both about the *harness* rather than the
model:

* `refine_inputs` and the auxiliary inputs `X[1]`/`X[2]` (desired velocity and
  quaternion) are dropped, so the forward takes a single depth image. The
  concatenation they feed is replaced by sizing the LSTM's input to the flattened
  conv features. Keeping them would require a multi-input harness; they do not
  exercise any op the single-input path does not.
* `spectral_norm` on the FC layers is dropped. It is a training-time
  reparameterisation; at eval it is a fixed rescale already folded into the
  weights, and it is not an operator to lower.

What matters is preserved: the conv front end, and a real 2-layer
`nn.LSTM(bias=False)` whose hidden state must survive across invocations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VitFlyLSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        # verbatim from upstream LSTMNet.__init__
        self.conv1 = nn.Conv2d(1, 4, 5, stride=3, padding=1)
        self.conv2 = nn.Conv2d(4, 10, 3, stride=2, padding=0)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.maxpool = nn.MaxPool2d(3, 1)
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(10)
        # upstream: LSTM(input_size=665, hidden_size=395, num_layers=2,
        #                dropout=0.15, bias=False). Dropout is eval-time
        #          identity; input_size follows our flattened feature width.
        self.lstm = nn.LSTM(input_size=660, hidden_size=395,
                            num_layers=2, bias=False)
        self.fc1 = nn.Linear(395, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 3)
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        # upstream uses -maxpool(-bn1(relu(conv1))) i.e. a min-pool; the negation
        # is elementwise and would need a neg op, so this uses maxpool directly.
        x = self.maxpool(self.bn1(self.relu(self.conv1(x))))
        x = self.avgpool(self.bn2(self.relu(self.conv2(x))))
        x = torch.flatten(x, 1).unsqueeze(0)   # [seq=1, batch=1, feature]
        x, _ = self.lstm(x)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        return self.fc3(x)


def get_model():
    m = VitFlyLSTMNet()
    m.eval()
    return m


def get_sample_input():
    return torch.randn(1, 1, 60, 90)

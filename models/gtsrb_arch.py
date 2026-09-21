"""SignNet model definition (PyTorch) — GTSRB street-sign classifier.

Vendored from the standalone gtsrb_signclf/model.py so this submodule doesn't
depend on a specific parent-repo layout to find its own architecture. Keep in
sync if the upstream copy changes.

Compact standard-conv CNN for GTSRB, envelope-matched to the KU040 Gemmini.
All convs are plain 3x3 (NO depthwise), BN + ReLU, global-avg-pool + FC head.
FX-traceable so modelblaster.pipeline.extract_graph can quantize it faithfully.

SignNetSoftmax is the DEPLOY wrapper: base logits + a softmax head so the int8
flow exercises softmax_s8 (this is the module the standalone eval_int8.py
quantized to reach float 99.0% / int8 98.97%, STOP recall 1.0).
"""

import torch
import torch.nn as nn


class SignNet(nn.Module):
    def __init__(self, in_ch=1, n_classes=43, input_size=48,
                 widths=(32, 64, 128)):
        super().__init__()
        c0, c1, c2 = widths
        # block 1
        self.conv1a = nn.Conv2d(in_ch, c0, 3, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(c0)
        self.conv1b = nn.Conv2d(c0, c0, 3, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(c0)
        self.pool1 = nn.MaxPool2d(2, 2)
        # block 2
        self.conv2a = nn.Conv2d(c0, c1, 3, padding=1, bias=False)
        self.bn2a = nn.BatchNorm2d(c1)
        self.conv2b = nn.Conv2d(c1, c1, 3, padding=1, bias=False)
        self.bn2b = nn.BatchNorm2d(c1)
        self.pool2 = nn.MaxPool2d(2, 2)
        # block 3
        self.conv3a = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.bn3a = nn.BatchNorm2d(c2)
        self.conv3b = nn.Conv2d(c2, c2, 3, padding=1, bias=False)
        self.bn3b = nn.BatchNorm2d(c2)
        self.pool3 = nn.MaxPool2d(2, 2)
        # head: global avg pool (fixed window) -> flatten -> FC
        feat = input_size // 8  # three /2 pools; 48 -> 6
        self.gap = nn.AvgPool2d(feat)
        self.fc = nn.Linear(c2, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))
        x = self.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2a(self.conv2a(x)))
        x = self.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class SignNetSoftmax(nn.Module):
    """Deploy wrapper: base logits + softmax head (exercises softmax_s8)."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        x = self.base(x)
        return torch.softmax(x, dim=1)

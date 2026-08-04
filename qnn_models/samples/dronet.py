"""DroNet model definition (PyTorch).

Copied from merlin/models/dronet/dronet.py with SMALL flag parameterized.
"""

import numpy as np
import torch
import torch.nn as nn


class DronetTorch(nn.Module):
    def __init__(self, img_dims, img_channels, output_dim, small=True):
        super().__init__()
        self.small = small
        self.conv_modules = nn.ModuleList()

        if small:
            self.conv_modules.append(nn.Conv2d(img_channels, 32, (3, 3), stride=(2, 2), padding=(1, 1)))
        else:
            self.conv_modules.append(nn.Conv2d(img_channels, 32, (5, 5), stride=(2, 2), padding=(2, 2)))

        filter_amt = np.array([32, 64, 128])
        for f in filter_amt:
            x1 = int(f / 2) if f != 32 else f
            x2 = f
            self.conv_modules.append(nn.Conv2d(x1, x2, (3, 3), stride=(2, 2), padding=(1, 1)))
            self.conv_modules.append(nn.Conv2d(x2, x2, (3, 3), padding=(1, 1)))
            self.conv_modules.append(nn.Conv2d(x1, x2, (1, 1), stride=(2, 2)))

        self.maxpool1 = nn.MaxPool2d((3, 3), (2, 2))

        bn_amt = np.array([32, 32, 32, 64, 64, 128])
        self.bn_modules = nn.ModuleList()
        for i in range(6):
            self.bn_modules.append(nn.BatchNorm2d(bn_amt[i]))

        self.relu_modules = nn.ModuleList()
        for i in range(7):
            self.relu_modules.append(nn.ReLU())

        self.dropout1 = nn.Dropout()

        linear_in = 2048 if small else 6272
        self.linear1 = nn.Linear(linear_in, output_dim)
        self.linear2 = nn.Linear(linear_in, output_dim)
        self.sigmoid1 = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.kaiming_normal_(self.conv_modules[1].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[2].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[4].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[5].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[7].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[8].weight)

    def forward(self, x):
        bn_idx = 0
        conv_idx = 1
        relu_idx = 0

        x = self.conv_modules[0](x)
        x1 = self.maxpool1(x)

        for i in range(3):
            x2 = self.bn_modules[bn_idx](x1)
            x2 = self.relu_modules[relu_idx](x2)
            x2 = self.conv_modules[conv_idx](x2)
            x2 = self.bn_modules[bn_idx + 1](x2)
            x2 = self.relu_modules[relu_idx + 1](x2)
            x2 = self.conv_modules[conv_idx + 1](x2)
            x1 = self.conv_modules[conv_idx + 2](x1)
            x3 = torch.add(x1, x2)
            x1 = x3
            bn_idx += 2
            relu_idx += 2
            conv_idx += 3

        x4 = torch.flatten(x3, start_dim=1)
        x4 = self.relu_modules[-1](x4)
        x5 = self.dropout1(x4)

        steer = self.linear1(x5)
        collision = self.linear2(x5)
        collision = self.sigmoid1(collision)

        return steer, collision

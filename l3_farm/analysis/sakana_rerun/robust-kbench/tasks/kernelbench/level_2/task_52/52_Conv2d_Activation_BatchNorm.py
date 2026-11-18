import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a convolution, applies activation, and then applies Batch Normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        # Add noise to match functional implementation
        self.conv.bias = nn.Parameter(self.conv.bias + torch.randn(self.conv.bias.shape) * 0.02)
        self.bn.bias = nn.Parameter(self.bn.bias + torch.randn(self.bn.bias.shape) * 0.02)
        self.bn.running_mean = self.bn.running_mean + torch.randn(self.bn.running_mean.shape) * 0.02
        self.bn.running_var = self.bn.running_var + torch.randn(self.bn.running_var.shape).abs() * 0.02

    def forward(self, x):
        x = self.conv(x)
        x = torch.multiply(torch.tanh(torch.nn.functional.softplus(x)), x)
        x = self.bn(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
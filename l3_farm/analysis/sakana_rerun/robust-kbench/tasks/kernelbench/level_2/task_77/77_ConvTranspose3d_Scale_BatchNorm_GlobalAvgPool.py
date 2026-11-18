import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, scales the output, applies batch normalization, 
    and then performs global average pooling. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.batch_norm.weight = nn.Parameter(self.batch_norm.weight + torch.randn(self.batch_norm.weight.shape)*0.02)
        self.batch_norm.bias = nn.Parameter(self.batch_norm.bias + torch.randn(self.batch_norm.bias.shape)*0.02)
        self.batch_norm.running_mean = self.batch_norm.running_mean + torch.randn(self.batch_norm.running_mean.shape)*0.02
        self.batch_norm.running_var = self.batch_norm.running_var + torch.randn(self.batch_norm.running_var.shape).abs()*0.02
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x * self.scale_factor
        x = self.batch_norm(x)
        x = self.global_avg_pool(x)
        return x

batch_size = 16
in_channels = 64
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, scales the result, and applies batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.gemm.bias = nn.Parameter(self.gemm.bias + torch.randn(self.gemm.bias.shape, device=self.gemm.bias.device, dtype=self.gemm.bias.dtype) * 0.02)
        self.scale = nn.Parameter(torch.randn(scale_shape) * 0.02)
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.bn.weight = nn.Parameter(self.bn.weight + torch.randn(self.bn.weight.shape, device=self.bn.weight.device, dtype=self.bn.weight.dtype) * 0.02)
        self.bn.bias = nn.Parameter(self.bn.bias + torch.randn(self.bn.bias.shape, device=self.bn.bias.device, dtype=self.bn.bias.dtype) * 0.02)
        self.bn.running_mean = torch.randn(out_features)
        self.bn.running_var = torch.abs(torch.randn(out_features))

    def forward(self, x):
        x = self.gemm(x)
        x = x * self.scale
        x = self.bn(x)
        return x

batch_size = 128
in_features = 1024
out_features = 512
scale_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scale_shape]
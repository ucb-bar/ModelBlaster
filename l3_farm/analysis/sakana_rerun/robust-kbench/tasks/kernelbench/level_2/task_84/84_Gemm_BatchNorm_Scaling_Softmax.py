import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), Batch Normalization, scaling, and Softmax.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        
        # Add noise to BatchNorm parameters and buffers
        self.bn.weight = nn.Parameter(self.bn.weight + torch.randn(self.bn.weight.shape)*0.02)
        self.bn.bias = nn.Parameter(self.bn.bias + torch.randn(self.bn.bias.shape)*0.02)
        self.bn.running_mean = self.bn.running_mean + torch.randn(self.bn.running_mean.shape)*0.02
        self.bn.running_var = self.bn.running_var + torch.randn(self.bn.running_var.shape).abs()*0.02
        
        # Initialize scale with noise instead of ones
        self.scale = nn.Parameter(torch.randn(scale_shape) * 0.02)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.bn(x)
        x = self.scale * x
        x = self.softmax(x)
        return x

batch_size = 128
in_features = 1024
out_features = 512
bn_eps = 1e-5
bn_momentum = 0.1
scale_shape = (1,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, scale_shape]
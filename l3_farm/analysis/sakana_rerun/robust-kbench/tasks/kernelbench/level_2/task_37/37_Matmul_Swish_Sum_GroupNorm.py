
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a matrix multiplication, applies Swish activation, sums with a bias term, and normalizes with GroupNorm.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.matmul.bias = nn.Parameter(self.matmul.bias + torch.randn(self.matmul.bias.shape, device=self.matmul.bias.device, dtype=self.matmul.bias.dtype) * 0.02)
        self.bias = nn.Parameter(torch.randn(bias_shape) * 0.02)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.group_norm.weight = nn.Parameter(self.group_norm.weight + torch.randn(self.group_norm.weight.shape, device=self.group_norm.weight.device, dtype=self.group_norm.weight.dtype) * 0.02)
        self.group_norm.bias = nn.Parameter(self.group_norm.bias + torch.randn(self.group_norm.bias.shape, device=self.group_norm.bias.device, dtype=self.group_norm.bias.dtype) * 0.02)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = torch.sigmoid(x) * x  # Swish activation
        x = x + self.bias
        x = self.group_norm(x)
        return x

batch_size = 128
in_features = 512
out_features = 1024
num_groups = 32
bias_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]

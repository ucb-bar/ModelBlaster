import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Performs matrix multiplication, adds bias, and applies ReLU activation.

    Args:
        x (torch.Tensor): Input tensor with shape (batch_size, in_features)
        weight (torch.Tensor): Weight matrix with shape (out_features, in_features)
        bias (torch.Tensor): Bias tensor with shape (out_features,)

    Returns:
        torch.Tensor: Output tensor with shape (batch_size, out_features)
    """
    x = F.linear(x, weight)
    x = x + bias
    x = F.relu(x)
    return x


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a bias term, and applies ReLU.
    """

    def __init__(self, in_features: int = 1024, out_features: int = 512):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features, bias=False)
        self.weight = nn.Parameter(gemm.weight)
        bias_shape = (out_features,)
        self.bias = nn.Parameter(torch.randn(bias_shape) * 0.02)

    def forward(self, x, fn=forward_fn):
        return fn(x, self.weight, self.bias)


def get_inputs(batch_size: int = 128, in_features: int = 1024):
    x = torch.randn(batch_size, in_features)
    return [x]



input_names = ['x']

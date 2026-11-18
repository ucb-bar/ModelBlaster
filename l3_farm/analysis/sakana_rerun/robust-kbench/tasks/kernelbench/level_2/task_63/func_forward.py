import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    divisor: float,
) -> torch.Tensor:
    """
    Applies linear transformation, ReLU activation, and division by constant.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features)
        bias (torch.Tensor): Bias vector of shape (out_features)
        divisor (float): Constant to divide by

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features)
    """
    x = F.linear(x, weight, bias)
    x = F.relu(x)
    x = x / divisor
    return x


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies ReLU, and divides by a constant.
    """

    def __init__(
        self,
        in_features: int = 1024,
        out_features: int = 512,
        divisor: float = 2.0,
    ):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(gemm.weight)
        self.bias = nn.Parameter(gemm.bias)
        self.divisor = divisor

    def forward(self, x, fn=forward_fn):
        return fn(x, self.weight, self.bias, self.divisor)


def get_inputs(batch_size: int = 128, in_features: int = 1024):
    x = torch.randn(batch_size, in_features)
    return [x]



input_names = ['x']

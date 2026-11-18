import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    constant: torch.Tensor,
) -> torch.Tensor:
    """
    Performs matrix multiplication, applies minimum with constant, and subtracts constant.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        linear_weight (torch.Tensor): Weight matrix of shape (out_features, in_features)
        linear_bias (torch.Tensor): Bias vector of shape (out_features)
        constant (torch.Tensor): Scalar constant tensor

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features)
    """
    x = F.linear(x, linear_weight, linear_bias)
    x = torch.min(x, constant)
    x = x - constant
    return x


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies minimum, and subtracts a constant.
    """

    def __init__(
        self, in_features: int = 10, out_features: int = 5, constant: float = 2.0
    ):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features)
        self.linear_weight = nn.Parameter(gemm.weight)
        self.linear_bias = nn.Parameter(gemm.bias)
        self.constant = nn.Parameter(torch.tensor(constant))

    def forward(self, x, fn=forward_fn):
        return fn(x, self.linear_weight, self.linear_bias, self.constant)


def get_inputs(batch_size: int = 128, in_features: int = 10):
    x = torch.randn(batch_size, in_features)
    return [x]



input_names = ['x']

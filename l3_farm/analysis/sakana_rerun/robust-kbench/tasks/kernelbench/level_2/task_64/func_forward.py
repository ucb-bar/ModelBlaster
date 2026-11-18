import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Performs matrix multiplication followed by LogSumExp, LeakyReLU, LeakyReLU, GELU, and GELU activations.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features)
        bias (torch.Tensor): Bias vector of shape (out_features)

    Returns:
        torch.Tensor: Output tensor after applying linear transformation and activations
    """
    # Gemm
    x = F.linear(x, weight, bias)
    # LogSumExp
    x = torch.logsumexp(x, dim=1, keepdim=True)
    # LeakyReLU
    x = F.leaky_relu(x, negative_slope=0.01)
    # LeakyReLU
    x = F.leaky_relu(x, negative_slope=0.01)
    # GELU
    x = F.gelu(x)
    # GELU
    x = F.gelu(x)
    return x


class Model(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), followed by LogSumExp, LeakyReLU,
    LeakyReLU, GELU, and GELU activations.
    """

    def __init__(self, in_features: int = 1024, out_features: int = 512):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features)
        self.weight = gemm.weight
        self.bias = gemm.bias

    def forward(self, x, fn=forward_fn):
        return fn(x, self.weight, self.bias)


def get_inputs(batch_size: int = 128, in_features: int = 1024):
    x = torch.randn(batch_size, in_features)
    return [x]



input_names = ['x']

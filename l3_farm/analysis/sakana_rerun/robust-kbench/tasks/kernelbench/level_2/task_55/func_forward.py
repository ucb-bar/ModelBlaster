import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    kernel_size: int,
    scale_factor: float,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Performs matrix multiplication, max pooling, sum, and scaling.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        kernel_size (int): Size of max pooling kernel
        scale_factor (float): Factor to scale the output by
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features)
        bias (torch.Tensor): Bias vector of shape (out_features)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size,)
    """
    x = F.linear(x, weight, bias)
    x = F.max_pool1d(x.unsqueeze(1), kernel_size).squeeze(1)
    x = torch.sum(x, dim=1)
    x = x * scale_factor
    return x


class Model(nn.Module):
    """
    Model that performs matrix multiplication, max pooling, sum, and scaling.
    """

    def __init__(
        self,
        in_features: int = 10,
        out_features: int = 5,
        kernel_size: int = 2,
        scale_factor: float = 0.5,
    ):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(gemm.weight)
        self.bias = nn.Parameter(gemm.bias)
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

    def forward(self, x, fn=forward_fn):
        return fn(x, self.kernel_size, self.scale_factor, self.weight, self.bias)


def get_inputs(batch_size: int = 128, in_features: int = 10):
    x = torch.randn(batch_size, in_features)
    return [x]



input_names = ['x']

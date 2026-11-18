import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    scale_factor: float,
    clamp_min: float,
    clamp_max: float,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies matrix multiplication, scaling, residual connection, clamping, LogSumExp and Mish activation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, input_size)
        scale_factor (float): Factor to scale the output by
        clamp_min (float): Minimum value for clamping
        clamp_max (float): Maximum value for clamping
        weight (torch.Tensor): Weight matrix of shape (hidden_size, input_size)
        bias (torch.Tensor): Bias vector of shape (hidden_size)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, hidden_size)
    """
    x = F.linear(x, weight, bias)
    x = x * scale_factor
    x = x + x
    x = torch.clamp(x, clamp_min, clamp_max)
    x = torch.logsumexp(x, dim=1, keepdim=True)
    x = x * F.mish(x)
    return x


class Model(nn.Module):
    """
    Model that performs a matrix multiplication, scales the result, adds a residual connection, clamps the output,
    applies LogSumExp, and finally applies the Mish activation function.
    """

    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 1024,
        scale_factor: float = 2.0,
        clamp_min: float = -10.0,
        clamp_max: float = 10.0,
    ):
        super(Model, self).__init__()
        matmul = nn.Linear(input_size, hidden_size)
        self.weight = matmul.weight
        self.bias = nn.Parameter(
            matmul.bias + torch.ones_like(matmul.bias) * 0.02
        )  # make sure its nonzero
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x, fn=forward_fn):
        return fn(
            x, self.scale_factor, self.clamp_min, self.clamp_max, self.weight, self.bias
        )


def get_inputs(batch_size: int = 128, input_size: int = 512):
    x = torch.randn(batch_size, input_size)
    return [x]



input_names = ['x']

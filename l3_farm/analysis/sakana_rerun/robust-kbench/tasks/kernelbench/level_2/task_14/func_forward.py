import torch
import torch.nn as nn


def forward_fn(
    x: torch.Tensor,
    scaling_factor: float,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    Performs matrix multiplication, division, summation and scaling.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, input_size)
        scaling_factor (float): Factor to scale the output by
        weight (torch.Tensor): Weight matrix of shape (hidden_size, input_size)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 1)
    """
    x = torch.matmul(x, weight.T)  # Gemm
    x = x / 2  # Divide
    x = torch.sum(x, dim=1, keepdim=True)  # Sum
    x = x * scaling_factor  # Scaling
    return x


class Model(nn.Module):
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    """

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 20,
        scaling_factor: float = 1.5,
    ):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size) * 0.02)
        self.scaling_factor = scaling_factor

    def forward(self, x, fn=forward_fn):
        return fn(x, self.scaling_factor, self.weight)


def get_inputs(batch_size: int = 128, input_size: int = 10):
    x = torch.randn(batch_size, input_size)
    return [x]


input_names = ["x"]

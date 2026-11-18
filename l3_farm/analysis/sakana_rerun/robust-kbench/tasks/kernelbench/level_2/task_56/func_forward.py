import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Performs matrix multiplication, applies sigmoid, and sums the result.

    Args:
        x: Input tensor of shape (batch_size, input_size)
        weight: Weight tensor of shape (hidden_size, input_size)
        bias: Bias tensor of shape (hidden_size)

    Returns:
        Output tensor of shape (batch_size, 1)
    """
    x = F.linear(x, weight, bias)
    x = torch.sigmoid(x)
    x = torch.sum(x, dim=1, keepdim=True)
    return x


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies sigmoid, and sums the result.
    """

    def __init__(self, input_size: int = 10, hidden_size: int = 20):
        super(Model, self).__init__()
        gemm = nn.Linear(input_size, hidden_size)
        self.weight = nn.Parameter(gemm.weight)
        self.bias = nn.Parameter(gemm.bias)

    def forward(self, x, fn=forward_fn):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        return fn(x, self.weight, self.bias)


def get_inputs(batch_size: int = 128, input_size: int = 10):
    x = torch.randn(batch_size, input_size)
    return [x]



input_names = ['x']

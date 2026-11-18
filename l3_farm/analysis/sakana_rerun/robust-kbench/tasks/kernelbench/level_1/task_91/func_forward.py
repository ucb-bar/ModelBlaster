import torch
import torch.nn as nn


def forward_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Performs a reverse cumulative sum operation.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): The dimension along which to perform the reverse cumulative sum.

    Returns:
        torch.Tensor: Output tensor.
    """
    return torch.cumsum(x.flip(dim), dim=dim).flip(dim)


class Model(nn.Module):
    """
    A model that performs a reverse cumulative sum operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim: int = 1):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x, fn=forward_fn):
        return fn(x, self.dim)


def get_inputs(batch_size: int = 128, input_shape: int = 4000):
    x = torch.randn(batch_size, *(input_shape,))
    return [x]


input_names = ["x"]

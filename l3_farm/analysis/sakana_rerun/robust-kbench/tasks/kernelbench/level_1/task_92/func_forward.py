import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Performs an exclusive cumulative sum (does not include the current element).

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): The dimension along which to perform the exclusive cumulative sum.

    Returns:
        torch.Tensor: Output tensor.
    """
    exclusive_cumsum = torch.cat(
        (torch.zeros_like(x.select(dim, 0).unsqueeze(dim)), x), dim=dim
    )[:-1]
    return torch.cumsum(exclusive_cumsum, dim=dim)


class Model(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim: int = 1):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x, fn=forward_fn):
        return fn(x, self.dim)


batch_size = 128
input_shape = (4000,)
dim = 1


def get_inputs(batch_size: int = 128, input_shape: int = 4000):
    x = torch.randn(batch_size, *(input_shape,))
    return [x]


input_names = ['x']

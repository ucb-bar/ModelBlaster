import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Performs a masked cumulative sum operation.
    Args:
        x (torch.Tensor): Input tensor.
        mask (torch.Tensor): Boolean mask tensor.
        dim (int): The dimension along which to perform the cumulative sum.

    Returns:
        torch.Tensor: Output tensor.
    """
    return torch.cumsum(x * mask, dim=dim)


class Model(nn.Module):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.
    """

    def __init__(self, dim: int = 1):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x, mask, fn=forward_fn):
        return fn(x, mask, self.dim)


def get_inputs(batch_size: int = 128, input_shape: int = 4000):
    x = torch.randn(batch_size, *(input_shape,))
    mask = torch.randint(0, 2, x.shape).bool()  # Random boolean mask
    return [x, mask]


input_names = ['x', 'mask']

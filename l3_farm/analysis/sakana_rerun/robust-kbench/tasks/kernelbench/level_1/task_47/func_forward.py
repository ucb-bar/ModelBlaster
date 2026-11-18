import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Applies sum reduction over the specified dimension.

    Args:
        x (torch.Tensor): Input tensor of shape (..., dim, ...).
        dim (int): Dimension to reduce over.

    Returns:
        torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
    """
    return torch.sum(x, dim=dim, keepdim=True)


class Model(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
    """

    def __init__(self, dim: int = 1):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        return fn(x, self.dim)


def get_inputs(batch_size: int = 16, dim1: int = 256, dim2: int = 256):
    x = torch.randn(batch_size, dim1, dim2)
    return [x]



input_names = ['x']

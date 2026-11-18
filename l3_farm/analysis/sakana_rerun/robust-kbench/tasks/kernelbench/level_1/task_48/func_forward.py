import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Reduces the input tensor along the specified dimension by taking the mean.

    Args:
        x (torch.Tensor): Input tensor of arbitrary shape.
        dim (int): The dimension to reduce over.

    Returns:
        torch.Tensor: Output tensor with reduced dimension. The shape of the output is the same as the input except for the reduced dimension which is removed.
    """
    return torch.mean(x, dim=dim)


class Model(nn.Module):
    """
    Simple model that performs mean reduction over a specific dimension.
    """

    def __init__(self, dim: int = 1):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Reduces the input tensor along the specified dimension by taking the mean.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension. The shape of the output is the same as the input except for the reduced dimension which is removed.
        """
        return fn(x, self.dim)


def get_inputs(batch_size: int = 16, dim1: int = 256, dim2: int = 256):
    x = torch.randn(batch_size, dim1, dim2)
    return [x]



input_names = ['x']

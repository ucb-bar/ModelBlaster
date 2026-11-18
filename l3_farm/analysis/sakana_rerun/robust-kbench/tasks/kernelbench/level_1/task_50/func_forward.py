import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Performs product reduction over the specified dimension.

    Args:
        x (torch.Tensor): Input tensor
        dim (int): Dimension to reduce over

    Returns:
        torch.Tensor: Output tensor with product reduction applied
    """
    return torch.prod(x, dim=dim)


class Model(nn.Module):
    """
    Simple model that performs product reduction over a dimension.
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
        Performs product reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor
            fn (callable): Function to use for forward pass

        Returns:
            torch.Tensor: Output tensor with product reduction applied
        """
        return fn(x, self.dim)


def get_inputs(batch_size: int = 16, dim1: int = 256, dim2: int = 256):
    x = torch.randn(batch_size, dim1, dim2)
    return [x]


input_names = ["x"]

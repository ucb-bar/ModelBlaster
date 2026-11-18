import torch
import torch.nn as nn
import torch.functional as F


def forward_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Applies argmax over the specified dimension to the input tensor.

    Args:
        x (torch.Tensor): Input tensor
        dim (int): Dimension to perform argmax over

    Returns:
        torch.Tensor: Output tensor with argmax applied over specified dimension
    """
    return torch.argmax(x, dim)


class Model(nn.Module):
    """
    Simple model that performs Argmax over a specified dimension.
    """

    def __init__(self, dim: int = 1):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Applies argmax over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor
            fn: Function to apply (defaults to forward_fn)

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        return fn(x, self.dim)


def get_inputs(batch_size: int = 16, dim1: int = 256, dim2: int = 256):
    x = torch.randn(batch_size, dim1, dim2)
    return [x]



input_names = ['x']

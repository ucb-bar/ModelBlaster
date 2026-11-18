import torch
import torch.nn as nn


def forward_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Applies min reduction over the specified dimension to the input tensor.

    Args:
        x (torch.Tensor): Input tensor
        dim (int): The dimension to reduce over

    Returns:
        torch.Tensor: Output tensor after min reduction over the specified dimension
    """
    return torch.min(x, dim)[0]


class Model(nn.Module):
    """
    Simple model that performs min reduction over a specific dimension.
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
        Applies min reduction over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor
            fn: Function to apply (defaults to forward_fn)

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension
        """
        return fn(x, self.dim)


def get_inputs(batch_size: int = 16, dim1: int = 256, dim2: int = 256):
    x = torch.randn(batch_size, dim1, dim2)
    return [x]


input_names = ["x"]

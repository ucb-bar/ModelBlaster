import torch
import torch.nn as nn


def forward_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Finds the index of the minimum value along the specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to find the minimum value.

    Returns:
        torch.Tensor: Tensor containing the indices of the minimum values along the specified dimension.
    """
    return torch.argmin(x, dim)


class Model(nn.Module):
    """
    Simple model that finds the index of the minimum value along a specified dimension.
    """

    def __init__(self, dim: int = 1):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x, fn=forward_fn):
        """
        Finds the index of the minimum value along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor.
            fn (callable): Function to compute the output. Defaults to forward_fn.

        Returns:
            torch.Tensor: Tensor containing the indices of the minimum values along the specified dimension.
        """
        return fn(x, self.dim)


def get_inputs(batch_size: int = 16, dim1: int = 256, dim2: int = 256):
    x = torch.randn(batch_size, dim1, dim2)
    return [x]



input_names = ['x']

import torch
import torch.nn as nn


def forward_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Performs a cumulative product operation.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): The dimension along which to perform the cumulative product.

    Returns:
        torch.Tensor: Output tensor.
    """
    return torch.cumprod(x, dim=dim)


class Model(nn.Module):
    """
    A model that performs a cumulative product operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim: int = 1):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x, fn=forward_fn):
        """
        Forward pass, computing the cumulative product along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        return fn(x, self.dim)


def get_inputs(batch_size: int = 128, input_shape: int = 4000):
    x = torch.randn(batch_size, *(input_shape,))
    return [x]


input_names = ["x"]

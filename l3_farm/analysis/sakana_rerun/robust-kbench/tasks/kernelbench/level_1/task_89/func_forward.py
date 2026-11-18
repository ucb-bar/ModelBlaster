import torch
import torch.nn as nn


def forward_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Performs a cumulative sum operation.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): The dimension along which to perform the cumulative sum.

    Returns:
        torch.Tensor: Output tensor.
    """
    return torch.cumsum(x, dim=dim)


class Model(nn.Module):
    """
    A simple model that performs a cumulative sum (prefix sum) operation along a specified dimension.
    """

    def __init__(self, dim: int = 1):
        """
        Initialize the Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x, fn=forward_fn):
        """
        Forward pass for the Scan model, computing the cumulative sum along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape)
            fn (callable): Function to compute the output, defaults to forward_fn
        """
        return fn(x, self.dim)


def get_inputs(batch_size: int = 128, input_shape: int = 4000):
    """
    Generates random inputs for testing the Scan model.

    Returns:
        list: A list containing a single randomly generated tensor with shape
              (batch_size, *input_shape).
    """
    x = torch.randn(batch_size, *(input_shape,))
    return [x]


input_names = ["x"]

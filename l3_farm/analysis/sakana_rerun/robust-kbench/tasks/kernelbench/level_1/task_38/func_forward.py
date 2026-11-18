import torch
import torch.nn as nn


def forward_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Applies L1 normalization to the input tensor using functional operations.

    Args:
        x (torch.Tensor): Input tensor of shape (..., dim, ...)

    Returns:
        torch.Tensor: Output tensor with L1 normalization applied, same shape as input
    """
    return x / torch.sum(torch.abs(x), dim=1, keepdim=True)


class Model(nn.Module):
    """
    Simple model that performs L1 normalization.
    """

    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...)
            fn: Function to apply (defaults to forward_fn)

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input
        """
        return fn(x)


def get_inputs(batch_size: int = 16, dim: int = 16384):
    x = torch.randn(batch_size, dim)
    return [x]



input_names = ['x']

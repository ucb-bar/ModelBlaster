import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Applies Frobenius norm normalization to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of arbitrary shape.

    Returns:
        torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
    """
    norm = torch.norm(x, p="fro")
    return x / norm


class Model(nn.Module):
    """
    Simple model that performs Frobenius norm normalization.
    """

    def __init__(self):
        """
        Initializes the Frobenius norm normalization layer.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Applies Frobenius norm normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.
            fn (callable): Function to apply normalization, defaults to forward_fn

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        return fn(x)


def get_inputs(
    batch_size: int = 16, num_features: int = 64, dim1: int = 256, dim2: int = 256
):
    x = torch.randn(batch_size, num_features, dim1, dim2)
    return [x]



input_names = ['x']

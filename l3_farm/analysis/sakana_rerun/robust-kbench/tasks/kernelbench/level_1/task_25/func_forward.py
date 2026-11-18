import torch
import torch.nn as nn


def forward_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Applies Swish activation to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of any shape.

    Returns:
        torch.Tensor: Output tensor with Swish applied, same shape as input.
    """
    return x * torch.sigmoid(x)


class Model(nn.Module):
    """
    Simple model that performs a Swish activation.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        return fn(x)


def get_inputs(batch_size: int = 16, dim: int = 16384):
    x = torch.randn(batch_size, dim)
    return [x]



input_names = ['x']

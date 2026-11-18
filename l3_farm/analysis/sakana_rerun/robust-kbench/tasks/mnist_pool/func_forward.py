import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(x: torch.Tensor) -> torch.Tensor:
    """Implements a max pooling layer with kernel size 2:

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        weights (torch.Tensor): Weights matrix of shape (out_channels, in_channels, kernel_height, kernel_width)
        biases (torch.Tensor): Biases vector of shape (out_channels)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width)
    """
    # Apply max pooling
    x = F.max_pool2d(x, kernel_size=2)
    return x


class Model(nn.Module):
    """
    Simple model that performs Feedforward network block.
    """

    def __init__(self):
        """
        Initializes the Feedforward network block.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Forward pass that calls forward_fn.
        """
        return fn(x)


def get_inputs(
    batch_size: int = 64,
    in_channels: int = 64,
    height: int = 28,
    width: int = 28,
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


input_names = ["x"]

import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    divisor: float,
) -> torch.Tensor:
    """
    Applies convolution, division by constant, and LeakyReLU.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        conv_weight (torch.Tensor): Convolution weights of shape (out_channels, in_channels, kernel_size, kernel_size)
        conv_bias (torch.Tensor): Convolution bias of shape (out_channels)
        divisor (float): Constant to divide by

    Returns:
        torch.Tensor: Output tensor after convolution, division and LeakyReLU activation
    """
    x = F.conv2d(x, conv_weight, bias=conv_bias)
    x = x / divisor
    x = F.leaky_relu(x, negative_slope=0.01)
    return x


class Model(nn.Module):
    """
    Simple model that performs a convolution, divides by a constant, and applies LeakyReLU.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        divisor: float = 2.0,
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias)
        self.divisor = divisor

    def forward(self, x, fn=forward_fn):
        return fn(x, self.conv_weight, self.conv_bias, self.divisor)


def get_inputs(
    batch_size: int = 128, in_channels: int = 3, height: int = 32, width: int = 32
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]



input_names = ['x']

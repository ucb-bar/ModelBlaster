import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    multiplier: torch.Tensor,
) -> torch.Tensor:
    """
    Applies convolution, scalar multiplication, LeakyReLU and GELU.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        conv_weight (torch.Tensor): Convolution weights of shape (out_channels, in_channels, kernel_size, kernel_size)
        conv_bias (torch.Tensor): Convolution bias of shape (out_channels)
        multiplier (torch.Tensor): Learnable scalar of shape (out_channels, 1, 1)

    Returns:
        torch.Tensor: Output tensor after applying convolution, multiplication, LeakyReLU and GELU
    """
    x = F.conv2d(x, conv_weight, bias=conv_bias)
    x = x * multiplier
    x = F.leaky_relu(x)
    x = F.gelu(x)
    return x


class Model(nn.Module):
    """
    Model that performs a convolution, multiplies by a learnable scalar, applies LeakyReLU, and then GELU.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias)
        multiplier_shape = (out_channels, 1, 1)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape) * 0.02)

    def forward(self, x, fn=forward_fn):
        return fn(x, self.conv_weight, self.conv_bias, self.multiplier)


def get_inputs(
    batch_size: int = 128, in_channels: int = 3, height: int = 32, width: int = 32
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]



input_names = ['x']

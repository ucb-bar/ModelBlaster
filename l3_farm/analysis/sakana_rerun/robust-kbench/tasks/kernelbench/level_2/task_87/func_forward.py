import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    subtract_value_1: float,
    subtract_value_2: float,
) -> torch.Tensor:
    """
    Applies convolution, subtracts two values, and applies Mish activation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        conv_weight (torch.Tensor): Convolution weight tensor of shape
            (out_channels, in_channels, kernel_size, kernel_size)
        conv_bias (torch.Tensor): Convolution bias tensor of shape (out_channels)
        subtract_value_1 (float): First value to subtract
        subtract_value_2 (float): Second value to subtract

    Returns:
        torch.Tensor: Output tensor after applying convolution, subtractions and Mish activation
    """
    x = F.conv2d(x, conv_weight, bias=conv_bias)
    x = x - subtract_value_1
    x = x - subtract_value_2
    x = F.mish(x)
    return x


class Model(nn.Module):
    """
    Model that performs a convolution, subtracts two values, applies Mish activation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        subtract_value_1: float = 0.5,
        subtract_value_2: float = 0.2,
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_weight = conv.weight
        self.conv_bias = conv.bias
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.conv_weight,
            self.conv_bias,
            self.subtract_value_1,
            self.subtract_value_2,
        )


def get_inputs(
    batch_size: int = 128, in_channels: int = 3, height: int = 32, width: int = 32
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]



input_names = ['x']

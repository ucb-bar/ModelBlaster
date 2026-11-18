import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    constant_value: float,
    scaling_factor: float,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies convolution, min with constant, bias addition and scaling.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        constant_value (float): Value to take minimum with
        scaling_factor (float): Factor to multiply output by
        conv_weight (torch.Tensor): Convolution weights
        conv_bias (torch.Tensor): Convolution bias
        bias (torch.Tensor): Bias tensor to add of shape (out_channels, 1, 1)

    Returns:
        torch.Tensor: Output tensor after applying convolution, min, bias and scaling
    """
    x = F.conv2d(x, conv_weight, bias=conv_bias)
    x = torch.min(x, torch.tensor(constant_value))
    x = x + bias
    x = x * scaling_factor
    return x


class Model(nn.Module):
    """
    Simple model that performs a convolution, takes the minimum with a constant,
    adds a bias term, and multiplies by a scaling factor.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        constant_value: float = 0.5,
        scaling_factor: float = 2.0,
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias + torch.ones_like(conv.bias) * 0.02)
        bias_shape = (out_channels, 1, 1)
        self.bias = nn.Parameter(torch.randn(bias_shape) * 0.02)
        self.constant_value = constant_value
        self.scaling_factor = scaling_factor

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.constant_value,
            self.scaling_factor,
            self.conv_weight,
            self.conv_bias,
            self.bias,
        )


def get_inputs(
    batch_size: int = 128, in_channels: int = 3, height: int = 32, width: int = 32
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]



input_names = ['x']

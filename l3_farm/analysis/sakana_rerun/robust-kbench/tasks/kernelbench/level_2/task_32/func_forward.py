import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    scale_factor: float,
) -> torch.Tensor:
    """
    Applies convolution, scales the output, and performs minimum operation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        conv_weight (torch.Tensor): Convolution weight tensor
        conv_bias (torch.Tensor): Convolution bias tensor
        scale_factor (float): Scale factor to multiply output by

    Returns:
        torch.Tensor: Output tensor after convolution, scaling and min operation
    """
    x = F.conv2d(x, conv_weight, bias=conv_bias)
    x = x * scale_factor
    x = torch.min(x, dim=1, keepdim=True)[0]  # Minimum along channel dimension
    return x


class Model(nn.Module):
    """
    Model that performs a convolution, scales the output, and then applies a minimum operation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        scale_factor: float = 2.0,
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias + torch.ones_like(conv.bias) * 0.02)
        self.scale_factor = scale_factor

    def forward(self, x, fn=forward_fn):
        return fn(x, self.conv_weight, self.conv_bias, self.scale_factor)


def get_inputs(
    batch_size: int = 128, in_channels: int = 3, height: int = 32, width: int = 32
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]



input_names = ['x']

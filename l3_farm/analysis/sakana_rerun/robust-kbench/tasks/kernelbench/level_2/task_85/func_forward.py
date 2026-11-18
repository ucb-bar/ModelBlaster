import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    scale: torch.Tensor,
    num_groups: int,
    maxpool_kernel_size: int,
    clamp_min: float,
    clamp_max: float,
) -> torch.Tensor:
    """
    Applies convolution, group normalization, scaling, max pooling and clamping.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        conv_weight (torch.Tensor): Convolution weights
        conv_bias (torch.Tensor): Convolution bias
        group_norm_weight (torch.Tensor): Group norm weights
        group_norm_bias (torch.Tensor): Group norm bias
        scale (torch.Tensor): Scale parameter of shape (out_channels, 1, 1)
        num_groups (int): Number of groups for group norm
        maxpool_kernel_size (int): Kernel size for max pooling
        clamp_min (float): Minimum value for clamping
        clamp_max (float): Maximum value for clamping

    Returns:
        torch.Tensor: Output tensor after applying all operations
    """
    x = F.conv2d(x, conv_weight, bias=conv_bias)
    x = F.group_norm(x, num_groups, weight=group_norm_weight, bias=group_norm_bias)
    x = x * scale
    x = F.max_pool2d(x, kernel_size=maxpool_kernel_size)
    x = torch.clamp(x, clamp_min, clamp_max)
    return x


class Model(nn.Module):
    """
    Model that performs convolution, group normalization, scaling, max pooling, and clamping.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        num_groups: int = 8,
        maxpool_kernel_size: int = 2,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias)
        group_norm = nn.GroupNorm(num_groups, out_channels)
        self.group_norm_weight = nn.Parameter(
            group_norm.weight + torch.randn(group_norm.weight.shape) * 0.02
        )
        self.group_norm_bias = nn.Parameter(
            group_norm.bias + torch.randn(group_norm.bias.shape) * 0.02
        )
        scale_shape = (out_channels, 1, 1)
        self.scale = nn.Parameter(torch.randn(scale_shape) * 0.02)
        self.num_groups = num_groups
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.conv_weight,
            self.conv_bias,
            self.group_norm_weight,
            self.group_norm_bias,
            self.scale,
            self.num_groups,
            self.maxpool_kernel_size,
            self.clamp_min,
            self.clamp_max,
        )


def get_inputs(
    batch_size: int = 128, in_channels: int = 3, height: int = 32, width: int = 32
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]



input_names = ['x']

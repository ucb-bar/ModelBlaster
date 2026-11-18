import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Applies 3D convolution, group normalization, and computes mean.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        conv_weight (torch.Tensor): 3D convolution weight tensor
        conv_bias (torch.Tensor): 3D convolution bias tensor
        group_norm_weight (torch.Tensor): Group norm weight tensor
        group_norm_bias (torch.Tensor): Group norm bias tensor
        num_groups (int): Number of groups for group normalization

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 1)
    """
    x = F.conv3d(x, conv_weight, bias=conv_bias)
    x = F.group_norm(x, num_groups, weight=group_norm_weight, bias=group_norm_bias)
    x = x.mean(dim=[1, 2, 3, 4])
    return x


class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, computes the mean
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        num_groups: int = 8,
    ):
        super(Model, self).__init__()
        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        group_norm = nn.GroupNorm(num_groups, out_channels)
        self.conv_weight = conv.weight
        self.conv_bias = nn.Parameter(
            conv.bias + torch.ones_like(conv.bias) * 0.02
        )  # make sure its nonzero
        self.group_norm_weight = group_norm.weight
        self.group_norm_bias = nn.Parameter(
            group_norm.bias + torch.ones_like(group_norm.bias) * 0.02
        )  # make sure its nonzero
        self.num_groups = num_groups

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.conv_weight,
            self.conv_bias,
            self.group_norm_weight,
            self.group_norm_bias,
            self.num_groups,
        )


def get_inputs(
    batch_size: int = 128, in_channels: int = 3, D: int = 16, H: int = 32, W: int = 32
):
    x = torch.randn(batch_size, in_channels, D, H, W)
    return [x]



input_names = ['x']

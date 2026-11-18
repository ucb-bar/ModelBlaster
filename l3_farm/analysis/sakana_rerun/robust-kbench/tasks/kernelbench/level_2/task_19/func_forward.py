import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    stride: int,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Applies transposed convolution, GELU activation, and group normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        stride (int): Stride of the transposed convolution
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        group_norm_weight (torch.Tensor): Weight tensor for group normalization
        group_norm_bias (torch.Tensor): Bias tensor for group normalization
        num_groups (int): Number of groups for group normalization

    Returns:
        torch.Tensor: Output tensor after applying transposed convolution, GELU and group norm
    """
    x = F.conv_transpose2d(x, conv_transpose, bias=conv_transpose_bias, stride=stride)
    x = F.gelu(x)
    x = F.group_norm(
        x, num_groups=num_groups, weight=group_norm_weight, bias=group_norm_bias
    )
    return x


class Model(nn.Module):
    """
    Model that performs a transposed convolution, applies GELU, and normalizes with GroupNorm.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 64,
        kernel_size: int = 4,
        stride: int = 2,
        groups: int = 8,
        num_groups: int = 8,
    ):
        super(Model, self).__init__()
        conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride
        )
        group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.conv_transpose_parameter = conv_transpose.weight
        self.conv_transpose_bias = nn.Parameter(
            conv_transpose.bias + torch.ones_like(conv_transpose.bias) * 0.02
        )  # make sure its nonzero
        self.group_norm_weight = group_norm.weight
        self.group_norm_bias = nn.Parameter(
            group_norm.bias + torch.ones_like(group_norm.bias) * 0.02
        )  # make sure its nonzero
        self.stride = stride
        self.num_groups = num_groups

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.stride,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
            self.group_norm_weight,
            self.group_norm_bias,
            self.num_groups,
        )


def get_inputs(
    batch_size: int = 128, in_channels: int = 32, height: int = 32, width: int = 32
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


input_names = ["x"]

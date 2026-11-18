import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    groups: int,
    eps: float,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 3D transposed convolution, Swish activation, group normalization and HardSwish activation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        stride (int): Stride of the transposed convolution
        padding (int): Padding of the transposed convolution
        groups (int): Number of groups for group normalization
        eps (float): Epsilon value for group normalization
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        group_norm_weight (torch.Tensor): Weight tensor for group normalization
        group_norm_bias (torch.Tensor): Bias tensor for group normalization

    Returns:
        torch.Tensor: Output tensor after applying all operations
    """
    x = F.conv_transpose3d(
        x, conv_transpose, bias=conv_transpose_bias, stride=stride, padding=padding
    )
    x = torch.sigmoid(x) * x  # Swish activation
    x = F.group_norm(
        x, num_groups=groups, weight=group_norm_weight, bias=group_norm_bias, eps=eps
    )
    x = F.hardswish(x)
    return x


class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies Swish activation,
    group normalization, and then HardSwish activation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        groups: int = 4,
        eps: float = 1e-5,
    ):
        super(Model, self).__init__()
        conv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.conv_transpose_parameter = nn.Parameter(conv.weight)
        self.conv_transpose_bias = nn.Parameter(conv.bias)
        gn = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.group_norm_weight = nn.Parameter(gn.weight)
        self.group_norm_bias = nn.Parameter(gn.bias + torch.randn(out_channels) * 0.02)
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.eps = eps

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.stride,
            self.padding,
            self.groups,
            self.eps,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
            self.group_norm_weight,
            self.group_norm_bias,
        )


def get_inputs(
    batch_size: int = 128,
    in_channels: int = 3,
    depth: int = 16,
    height: int = 32,
    width: int = 32,
):
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]



input_names = ['x']

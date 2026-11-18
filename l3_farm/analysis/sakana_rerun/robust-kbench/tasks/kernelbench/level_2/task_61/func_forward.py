import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    conv_transpose: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    groups: int,
    eps: float,
) -> torch.Tensor:
    """
    Applies a transposed 3D convolution, ReLU, and group normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        group_norm_weight (torch.Tensor): Weight tensor for group normalization
        group_norm_bias (torch.Tensor): Bias tensor for group normalization
        groups (int): Number of groups for group normalization
        eps (float): Epsilon for group normalization
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W)
    """
    x = F.conv_transpose3d(x, conv_transpose, bias=None)
    x = F.relu(x)
    x = F.group_norm(x, groups, group_norm_weight, group_norm_bias, eps)
    return x


class Model(nn.Module):
    """
    Model that performs a transposed 3D convolution, applies ReLU, and then applies group normalization.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 128,
        kernel_size: int = 3,
        groups: int = 8,
        bias: bool = False,
        eps: float = 1e-5,
    ):
        super(Model, self).__init__()
        conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.conv_transpose_parameter = conv.weight
        gn = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.group_norm_weight = nn.Parameter(
            gn.weight + torch.randn_like(gn.weight) * 0.02
        )
        self.group_norm_bias = nn.Parameter(gn.bias + torch.randn_like(gn.bias) * 0.02)

        self.groups = groups
        self.eps = eps

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.conv_transpose_parameter,
            self.group_norm_weight,
            self.group_norm_bias,
            self.groups,
            self.eps,
        )


def get_inputs(
    batch_size: int = 16, in_channels: int = 64, D: int = 8, H: int = 16, W: int = 16
):
    x = torch.randn(batch_size, in_channels, D, H, W)
    return [x]


input_names = ["x"]

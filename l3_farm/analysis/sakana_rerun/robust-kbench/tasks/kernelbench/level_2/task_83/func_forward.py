import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    min_value: float,
    max_value: float,
    dropout_p: float,
    num_groups: int,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 3D convolution, Group Normalization, clamp and dropout operations.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        min_value (float): Minimum value for clamp operation
        max_value (float): Maximum value for clamp operation
        dropout_p (float): Dropout probability
        conv_weight (torch.Tensor): 3D convolution weights
        conv_bias (torch.Tensor): 3D convolution bias
        norm_weight (torch.Tensor): Group norm weights
        norm_bias (torch.Tensor): Group norm bias

    Returns:
        torch.Tensor: Output tensor after applying convolution, normalization, min, clamp and dropout
    """
    x = F.conv3d(x, conv_weight, conv_bias)
    x = F.group_norm(x, num_groups=num_groups, weight=norm_weight, bias=norm_bias)
    x = torch.clamp(x, min=min_value, max=max_value)
    x = F.dropout(x, p=dropout_p, training=True)
    return x


class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, minimum, clamp, and dropout.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        num_groups: int = 8,
        min_value: float = 0.0,
        max_value: float = 1.0,
        dropout_p: float = 0.2,
    ):
        super(Model, self).__init__()
        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        torch.manual_seed(0)
        group_norm = nn.GroupNorm(num_groups, out_channels)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias)
        self.norm_weight = nn.Parameter(
            group_norm.weight + torch.randn(group_norm.weight.shape) * 0.02
        )
        self.norm_bias = nn.Parameter(
            group_norm.bias + torch.randn(group_norm.bias.shape) * 0.02
        )
        self.min_value = min_value
        self.max_value = max_value
        self.dropout_p = dropout_p
        self.num_groups = num_groups

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.min_value,
            self.max_value,
            self.dropout_p,
            self.num_groups,
            self.conv_weight,
            self.conv_bias,
            self.norm_weight,
            self.norm_bias,
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

import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    multiplier: torch.Tensor,
    instance_norm_weight: torch.Tensor,
    instance_norm_bias: torch.Tensor,
    clamp_min: float,
    clamp_max: float,
) -> torch.Tensor:
    """
    Applies 3D convolution, multiplication, instance normalization, clamping, multiplication and max operation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        conv_weight (torch.Tensor): 3D convolution weights
        conv_bias (torch.Tensor): 3D convolution bias
        multiplier (torch.Tensor): Multiplier tensor of shape (out_channels, 1, 1, 1)
        instance_norm_weight (torch.Tensor): Instance norm weight
        instance_norm_bias (torch.Tensor): Instance norm bias
        clamp_min (float): Minimum value for clamping
        clamp_max (float): Maximum value for clamping

    Returns:
        torch.Tensor: Output tensor after applying operations
    """
    x = F.conv3d(x, conv_weight, conv_bias)
    x = x * multiplier
    x = F.instance_norm(x, instance_norm_weight, instance_norm_bias)
    x = torch.clamp(x, clamp_min, clamp_max)
    x = x * multiplier
    x = torch.max(x, dim=1)[0]
    return x


class Model(nn.Module):
    """
    A 3D convolutional layer followed by multiplication, instance normalization, clamping, multiplication, and a max operation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        clamp_min: float = -1.0,
        clamp_max: float = 1.0,
    ):
        super(Model, self).__init__()
        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias)
        multiplier_shape = (out_channels, 1, 1, 1)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))  # * 0.02)
        self.instance_norm_weight = nn.Parameter(
            torch.ones(out_channels) + torch.randn(out_channels) * 0.02
        )
        self.instance_norm_bias = nn.Parameter(
            torch.zeros(out_channels) + torch.randn(out_channels) * 0.02
        )
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.conv_weight,
            self.conv_bias,
            self.multiplier,
            self.instance_norm_weight,
            self.instance_norm_bias,
            self.clamp_min,
            self.clamp_max,
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

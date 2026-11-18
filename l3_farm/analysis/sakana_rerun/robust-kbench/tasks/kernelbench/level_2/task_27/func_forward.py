import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 3D convolution, HardSwish, ReLU, Softmax and mean reduction.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        conv_weight (torch.Tensor): 3D convolution weight tensor of shape
            (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        conv_bias (torch.Tensor): Bias tensor for 3D convolution of shape (out_channels)

    Returns:
        torch.Tensor: Output tensor after applying convolution, activations and reduction,
            with shape (batch_size, out_channels)
    """
    x = F.conv3d(x, conv_weight, bias=conv_bias)
    x = F.hardswish(x)
    x = F.relu(x)
    x = F.softmax(x, dim=1)
    x = torch.mean(x, dim=[2, 3, 4])
    return x


class Model(nn.Module):
    """
    Simple model that performs a 3D convolution, applies HardSwish, ReLU, Softmax, and then calculates the mean.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
    ):
        super(Model, self).__init__()

        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias + torch.ones_like(conv.bias) * 0.02)

    def forward(self, x, fn=forward_fn):
        return fn(x, self.conv_weight, self.conv_bias)


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

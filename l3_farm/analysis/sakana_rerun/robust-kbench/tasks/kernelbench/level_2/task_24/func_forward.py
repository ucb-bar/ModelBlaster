import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    dim: int,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 3D convolution, minimum operation along specified dimension, and softmax.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        dim (int): Dimension along which to apply minimum operation
        conv_weight (torch.Tensor): 3D convolution weight tensor
        conv_bias (torch.Tensor): 3D convolution bias tensor

    Returns:
        torch.Tensor: Output tensor after applying convolution, min and softmax
    """
    x = F.conv3d(x, conv_weight, bias=conv_bias)
    x = torch.min(x, dim=dim)[0]  # Apply minimum along the specified dimension
    x = F.softmax(x, dim=1)  # Apply softmax along the channel dimension
    return x


class Model(nn.Module):
    """
    Simple model that performs a 3D convolution, applies minimum operation along a specific dimension,
    and then applies softmax.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        dim: int = 2,
    ):
        super(Model, self).__init__()
        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.conv_weight = conv.weight
        self.conv_bias = nn.Parameter(
            conv.bias + torch.ones_like(conv.bias) * 0.02
        )  # make sure its nonzero
        self.dim = dim

    def forward(self, x, fn=forward_fn):
        return fn(x, self.dim, self.conv_weight, self.conv_bias)


def get_inputs(
    batch_size: int = 128, in_channels: int = 3, D: int = 16, H: int = 32, W: int = 32
):
    x = torch.randn(batch_size, in_channels, D, H, W)
    return [x]



input_names = ['x']

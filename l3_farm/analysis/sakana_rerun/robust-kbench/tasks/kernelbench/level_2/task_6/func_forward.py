import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
) -> torch.Tensor:
    """Applies 3D convolution, softmax activation, and two max pooling operations.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        conv_weight (torch.Tensor): Convolution weight tensor of shape
            (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        conv_bias (torch.Tensor): Bias tensor for convolution of shape (out_channels)

    Returns:
        torch.Tensor: Output tensor after applying convolution, softmax and max pooling,
            with shape (batch_size, out_channels, depth', height', width') where:
            depth' = ((depth - kernel_size + 1) // 4)
            height' = ((height - kernel_size + 1) // 4)
            width' = ((width - kernel_size + 1) // 4)
            The //4 comes from two max pooling operations with kernel_size=2
    """
    x = F.conv3d(x, conv_weight, conv_bias, stride=1, padding=0)
    x = F.softmax(x, dim=1)
    x = F.max_pool3d(x, kernel_size=2)
    x = F.max_pool3d(x, kernel_size=2)
    return x


class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Softmax, and performs two max pooling operations.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        pool_kernel_size: int = 2,
    ):
        super(Model, self).__init__()
        conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=1)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias)

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

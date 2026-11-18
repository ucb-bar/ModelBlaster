import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    divisor: float,
    pool_size: tuple,
    sum_dim: int,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 3D convolution, division, max pooling, global average pooling, bias addition and sum.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        divisor (float): Constant to divide by
        pool_size (tuple): Size for max pooling (depth, height, width)
        sum_dim (int): Dimension to sum over
        conv_weight (torch.Tensor): 3D convolution weights
        conv_bias (torch.Tensor): 3D convolution bias
        bias (torch.Tensor): Bias tensor for addition

    Returns:
        torch.Tensor: Output tensor after applying all operations
    """
    x = F.conv3d(x, conv_weight, bias=conv_bias)
    x = x / divisor
    x = F.max_pool3d(x, pool_size)
    x = F.adaptive_avg_pool3d(x, (1, 1, 1))
    x = x + bias
    x = torch.sum(x, dim=sum_dim)
    return x


class Model(nn.Module):
    """
    Model that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: tuple = (3, 3, 3),
        divisor: float = 2.0,
        pool_size: tuple = (2, 2, 2),
        sum_dim: int = 1,
    ):
        super(Model, self).__init__()
        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        bias_shape = (out_channels, 1, 1, 1)
        self.bias = nn.Parameter(torch.randn(bias_shape) * 0.02)
        self.conv_weight = conv.weight
        self.conv_bias = conv.bias
        self.divisor = divisor
        self.pool_size = pool_size
        self.sum_dim = sum_dim

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.divisor,
            self.pool_size,
            self.sum_dim,
            self.conv_weight,
            self.conv_bias,
            self.bias,
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

import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    pool_kernel_size: int = 2,
) -> torch.Tensor:
    """
    Performs convolution, average pooling, applies sigmoid, and sums the result.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        conv_weight (torch.Tensor): Convolution weights of shape (out_channels, in_channels, kernel_size, kernel_size)
        conv_bias (torch.Tensor): Convolution bias of shape (out_channels)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size,) containing summed values
    """
    x = F.conv2d(x, conv_weight, bias=conv_bias)
    x = F.avg_pool2d(x, pool_kernel_size)
    x = torch.sigmoid(x)
    x = torch.sum(x, dim=[1, 2, 3])
    return x


class Model(nn.Module):
    """
    This model performs a convolution, average pooling, applies sigmoid, and sums the result.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        pool_kernel_size: int = 2,
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x, fn=forward_fn):
        return fn(x, self.conv_weight, self.conv_bias, self.pool_kernel_size)


def get_inputs(
    batch_size: int = 128, in_channels: int = 3, height: int = 32, width: int = 32
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]



input_names = ['x']

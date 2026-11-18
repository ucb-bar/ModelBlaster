import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    scaling_factor: float,
    pool_kernel_size: int,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies convolution, tanh activation, scaling, bias addition and max pooling.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        scaling_factor (float): Factor to scale the tensor by after tanh
        pool_kernel_size (int): Size of max pooling kernel
        conv_weight (torch.Tensor): Convolution weights
        conv_bias (torch.Tensor): Convolution bias
        bias (torch.Tensor): Bias tensor for addition of shape (out_channels, 1, 1)

    Returns:
        torch.Tensor: Output tensor after applying convolution, tanh, scaling, bias and max pooling
    """
    x = F.conv2d(x, conv_weight, bias=conv_bias)
    x = torch.tanh(x)
    x = x * scaling_factor
    x = x + bias
    x = F.max_pool2d(x, pool_kernel_size)
    return x


class Model(nn.Module):
    """
    A model that performs a convolution, applies tanh, scaling, adds a bias term, and then max-pools.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        scaling_factor: float = 2.0,
        pool_kernel_size: int = 2,
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias)
        bias_shape = (out_channels, 1, 1)
        self.bias = nn.Parameter(torch.randn(bias_shape) * 0.02)
        self.scaling_factor = scaling_factor
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.scaling_factor,
            self.pool_kernel_size,
            self.conv_weight,
            self.conv_bias,
            self.bias,
        )


def get_inputs(
    batch_size: int = 128, in_channels: int = 3, height: int = 32, width: int = 32
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]



input_names = ['x']

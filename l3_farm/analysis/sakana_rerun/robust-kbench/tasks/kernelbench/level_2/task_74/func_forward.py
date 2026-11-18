import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    output_padding: int,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    multiplier: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 3D transposed convolution, LeakyReLU, multiplication, LeakyReLU and max pooling.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        stride (int): Stride for the transposed convolution
        padding (int): Padding for the transposed convolution
        output_padding (int): Output padding for the transposed convolution
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        multiplier (torch.Tensor): Multiplier tensor of shape (out_channels, 1, 1, 1)

    Returns:
        torch.Tensor: Output tensor after applying operations
    """
    x = F.conv_transpose3d(
        x,
        conv_transpose,
        bias=conv_transpose_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )
    x = F.leaky_relu(x, negative_slope=0.2)
    x = x * multiplier
    x = F.leaky_relu(x, negative_slope=0.2)
    x = F.max_pool3d(x, kernel_size=2)
    return x


class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies LeakyReLU, multiplies by a learnable parameter,
    applies LeakyReLU again, and performs a max pooling operation.
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 32,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
    ):
        super(Model, self).__init__()
        conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.conv_transpose_parameter = nn.Parameter(conv.weight)
        self.conv_transpose_bias = nn.Parameter(conv.bias)
        multiplier_shape = (out_channels, 1, 1, 1)
        self.multiplier_parameter = nn.Parameter(torch.randn(multiplier_shape) * 0.02)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.stride,
            self.padding,
            self.output_padding,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
            self.multiplier_parameter,
        )


def get_inputs(
    batch_size: int = 16,
    in_channels: int = 16,
    depth: int = 16,
    height: int = 32,
    width: int = 32,
):
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


input_names = ["x"]

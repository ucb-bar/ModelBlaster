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
    bias: torch.Tensor,
) -> torch.Tensor:
    """Applies transposed convolution, bias subtraction and tanh activation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        stride (int): Stride of the convolution
        padding (int): Zero-padding added to both sides of input
        output_padding (int): Additional size added to output shape
        conv_transpose (torch.Tensor): Transposed convolution weight tensor of shape
            (in_channels, out_channels, kernel_height, kernel_width)
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution of shape (out_channels)
        bias (torch.Tensor): Bias tensor to subtract of shape (out_channels, 1, 1)

    Returns:
        torch.Tensor: Output tensor after applying transposed convolution, bias subtraction and tanh,
            with shape (batch_size, out_channels, output_height, output_width)
            where output_height = stride * (height - 1) - 2 * padding + kernel_height + output_padding
            and output_width = stride * (width - 1) - 2 * padding + kernel_width + output_padding
    """
    x = F.conv_transpose2d(
        x,
        conv_transpose,
        bias=conv_transpose_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )
    x = x - bias
    x = torch.tanh(x)
    return x


class Model(nn.Module):
    """
    Model that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 16,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
    ):
        super(Model, self).__init__()
        conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.conv_transpose_parameter = nn.Parameter(conv_transpose.weight)
        self.conv_transpose_bias = nn.Parameter(conv_transpose.bias)
        bias_shape = (out_channels, 1, 1)
        self.bias_parameter = nn.Parameter(torch.randn(bias_shape) * 0.02)
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
            self.bias_parameter,
        )


def get_inputs(
    batch_size: int = 128, in_channels: int = 32, height: int = 16, width: int = 16
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


input_names = ["x"]

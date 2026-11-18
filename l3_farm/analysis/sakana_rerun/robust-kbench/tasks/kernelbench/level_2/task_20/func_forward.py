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
    """
    Applies a 3D transposed convolution followed by bias addition and residual operations.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        stride (int): Stride of the transposed convolution
        padding (int): Padding of the transposed convolution
        output_padding (int): Additional size added to output shape
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        bias (torch.Tensor): Bias tensor for addition

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
    original_x = x.clone().detach()
    x = x + bias
    x = x + original_x
    x = x * original_x
    x = x + original_x
    return x


class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum,
    a residual add, a multiplication, and another residual add.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
    ):
        super(Model, self).__init__()
        conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.conv_transpose_parameter = conv_transpose.weight
        self.conv_transpose_bias = nn.Parameter(
            conv_transpose.bias + torch.ones_like(conv_transpose.bias) * 0.02
        )  # make sure its nonzero
        self.bias_shape = (out_channels, 1, 1, 1)
        self.bias_parameter = nn.Parameter(torch.randn(self.bias_shape) * 0.02)
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
    batch_size: int = 16,
    in_channels: int = 32,
    depth: int = 16,
    height: int = 32,
    width: int = 32,
):
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


input_names = ["x"]

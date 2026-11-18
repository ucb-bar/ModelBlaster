import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    add_input: torch.Tensor,
    stride: int,
    padding: int,
    output_padding: int,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies a 3D transposed convolution operation followed by tensor addition and HardSwish activation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        add_input (torch.Tensor): Input tensor to be added after transposed convolution
        stride (int): Stride of the transposed convolution
        padding (int): Padding of the transposed convolution
        output_padding (int): Additional size added to output shape
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution

    Returns:
        torch.Tensor: Output tensor after applying transposed convolution, addition and HardSwish
    """
    x = F.conv_transpose3d(
        x,
        conv_transpose,
        bias=conv_transpose_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )
    x = x + add_input
    x = x * F.hardswish(x)
    return x


class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
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
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x, add_input, fn=forward_fn):
        return fn(
            x,
            add_input,
            self.stride,
            self.padding,
            self.output_padding,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
        )


def get_inputs(
    batch_size: int = 128,
    in_channels: int = 32,
    out_channels: int = 64,
    D: int = 16,
    H: int = 16,
    W: int = 16,
    stride: int = 2,
):
    x = torch.randn(batch_size, in_channels, D, H, W)
    add_input = torch.randn(
        batch_size, out_channels, D * stride, H * stride, W * stride
    )
    return [x, add_input]



input_names = ['x', 'add_input']

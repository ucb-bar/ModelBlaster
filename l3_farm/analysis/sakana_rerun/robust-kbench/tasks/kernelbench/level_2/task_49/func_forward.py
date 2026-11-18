import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    output_padding: int,
    bias_flag: bool,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies a 3D transposed convolution operation followed by softmax and sigmoid.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        stride (int): Stride of the transposed convolution
        padding (int): Padding of the transposed convolution
        output_padding (int): Additional size added to output shape
        bias_flag (bool): Whether to use bias in conv_transpose
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution

    Returns:
        torch.Tensor: Output tensor after applying transposed convolution, softmax and sigmoid
    """
    bias = conv_transpose_bias if bias_flag else None
    x = F.conv_transpose3d(
        x,
        conv_transpose,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )
    x = F.softmax(x, dim=1)
    x = torch.sigmoid(x)
    return x


class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
        bias: bool = True,
    ):
        super(Model, self).__init__()
        conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )
        self.conv_transpose_parameter = nn.Parameter(conv_transpose.weight)
        self.conv_transpose_bias = (
            nn.Parameter(
                conv_transpose.bias
                + torch.randn(
                    conv_transpose.bias.shape,
                    device=conv_transpose.bias.device,
                    dtype=conv_transpose.bias.dtype,
                )
                * 0.02
            )
            if bias
            else None
        )
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias = bias

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.stride,
            self.padding,
            self.output_padding,
            self.bias,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
        )


def get_inputs(
    batch_size: int = 16, in_channels: int = 32, D: int = 16, H: int = 32, W: int = 32
):
    x = torch.randn(batch_size, in_channels, D, H, W)
    return [x]



input_names = ['x']

import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    groups: int,
) -> torch.Tensor:
    """
    Performs a 2D transposed convolution operation with asymmetric input, asymmetric kernel, grouped, padded, and dilated.

    Args:
        x (torch.Tensor): Input tensor.
        weight (torch.Tensor): Weight tensor.
        bias (torch.Tensor): Bias tensor.
        stride (tuple): Stride of the convolution.
        padding (tuple): Padding applied to the input.
        dilation (tuple): Dilation of the convolution.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor.
    """
    return F.conv_transpose2d(
        x,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


class Model(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input, asymmetric kernel,
    grouped, padded, and dilated.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (tuple): Stride of the convolution (height, width).
        padding (tuple): Padding applied to the input (height, width).
        dilation (tuple): Spacing between kernel elements (height, width).
        groups (int): Number of blocked connections from input channels to output channels.
        bias (bool): If `True`, adds a learnable bias to the output.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 64,
        kernel_size: tuple = (3, 5),
        stride: tuple = (2, 3),
        padding: tuple = (1, 2),
        dilation: tuple = (2, 1),
        groups: int = 4,
        bias: bool = False,
    ):
        super(Model, self).__init__()
        conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # Copy the initialized parameters
        self.weight = nn.Parameter(conv.weight.clone())
        self.bias = nn.Parameter(conv.bias.clone()) if bias else None

        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def get_inputs(
    batch_size: int = 16, in_channels: int = 32, height: int = 128, width: int = 256
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]



input_names = ['x']

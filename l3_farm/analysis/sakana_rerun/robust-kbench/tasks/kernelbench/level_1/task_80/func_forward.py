import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: tuple,
    dilation: tuple,
) -> torch.Tensor:
    """
    Performs a standard 2D convolution operation with square input and asymmetric kernel, with dilation and padding.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        weight (torch.Tensor): Weight tensor of shape (out_channels, in_channels, kernel_height, kernel_width).
        bias (torch.Tensor): Bias tensor of shape (out_channels).
        stride (int): Stride of the convolution.
        padding (tuple): Padding applied to the input (top/bottom, left/right).
        dilation (tuple): Spacing between kernel elements (height, width).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
    """
    return F.conv2d(
        x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation
    )


class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with square input and asymmetric kernel, with dilation and padding.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        kernel_size: tuple = (3, 5),
        stride: int = 1,
        padding: tuple = (1, 2),
        dilation: tuple = (2, 1),
        bias: bool = False,
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        # Copy the initialized parameters
        self.weight = nn.Parameter(conv.weight.clone())
        self.bias = nn.Parameter(conv.bias.clone()) if bias else None

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(
        self,
        x: torch.Tensor,
        fn=forward_fn,
    ) -> torch.Tensor:
        return fn(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
        )


def get_inputs(
    batch_size: int = 16, in_channels: int = 3, height: int = 256, width: int = 256
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]



input_names = ['x']

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
    Implementation of 2D convolution with asymmetric kernel.

    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width).
        weight: Weight tensor of shape (out_channels, in_channels // groups, kernel_size[0], kernel_size[1]).
        bias: Bias tensor of shape (out_channels).
        stride: Stride of the convolution.
        padding: Padding of the convolution.
        dilation: Dilation of the convolution.
        groups: Number of groups in the convolution.

    Returns:
        Output tensor of shape (batch_size, out_channels, height, width).
    """
    return F.conv2d(
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
    Performs a standard 2D convolution operation with asymmetric input and kernel sizes.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        kernel_size: tuple = (3, 5),
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super(Model, self).__init__()
        # Create a Conv2d layer to get the same initialization
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
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
        self.dilation = dilation
        self.groups = groups

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
            self.groups,
        )


def get_inputs(
    batch_size: int = 16, in_channels: int = 3, height: int = 256, width: int = 128
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


input_names = ["x"]

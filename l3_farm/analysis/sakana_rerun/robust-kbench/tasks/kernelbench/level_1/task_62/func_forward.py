import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> torch.Tensor:
    """
    Performs a standard 2D convolution operation with a square input and an asymmetric kernel.

    Args:
        x (torch.Tensor): Input tensor.
        weight (torch.Tensor): Weight tensor.
        bias (torch.Tensor): Bias tensor.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        dilation (int): Dilation of the convolution.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor.
    """
    return F.conv2d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and an asymmetric kernel.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        kernel_size: tuple = (3, 5),
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
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
    batch_size: int = 16, in_channels: int = 3, height: int = 256, width: int = 256
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]



input_names = ['x']

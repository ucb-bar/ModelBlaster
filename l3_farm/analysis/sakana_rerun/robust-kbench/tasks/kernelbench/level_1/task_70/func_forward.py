import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    output_padding: tuple,
    dilation: tuple,
    groups: int,
) -> torch.Tensor:
    """
    Performs a transposed 3D convolution operation with asymmetric input and a square kernel.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).
        weight (torch.Tensor): Weight tensor of shape (out_channels, in_channels, kernel_size, kernel_size, kernel_size).
        bias (torch.Tensor): Bias tensor of shape (out_channels).
        stride (tuple): Stride of the convolution.
        padding (tuple): Padding applied to the input.
        output_padding (tuple): Additional size added to one side of each dimension in the output shape.
        dilation (tuple): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
    """
    return F.conv_transpose3d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )


class Model(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and a square kernel.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 16,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(Model, self).__init__()
        conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
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
        self.output_padding = output_padding
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
            self.output_padding,
            self.dilation,
            self.groups,
        )


def get_inputs(
    batch_size: int = 16,
    in_channels: int = 32,
    depth: int = 16,
    height: int = 32,
    width: int = 64,
):
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]



input_names = ['x']

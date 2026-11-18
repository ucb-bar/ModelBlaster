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
    groups: int,
) -> torch.Tensor:
    """
    Performs the transposed 3D convolution using functional interface.

    Args:
        x (torch.Tensor): Input tensor.
        weight (torch.Tensor): Weight tensor.
        bias (torch.Tensor): Bias tensor.
        stride (tuple): Stride for the convolution.
        padding (tuple): Padding for the convolution.
        output_padding (tuple): Output padding for the convolution.
        groups (int): Number of groups for the convolution.

    Returns:
        torch.Tensor: Output tensor after convolution.
    """
    return F.conv_transpose3d(
        x,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
    )


class Model(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and kernel sizes.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 16,
        kernel_size: tuple = (3, 5, 7),
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
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
            self.groups,
        )


def get_inputs(
    batch_size: int = 16,
    in_channels: int = 32,
    depth_in: int = 16,
    height_in: int = 32,
    width_in: int = 64,
):
    x = torch.randn(batch_size, in_channels, depth_in, height_in, width_in)
    return [x]


input_names = ["x"]

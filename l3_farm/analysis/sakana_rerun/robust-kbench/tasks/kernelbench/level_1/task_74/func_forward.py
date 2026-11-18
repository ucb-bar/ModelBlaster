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
) -> torch.Tensor:
    """
    Performs the transposed 1D convolution using functional interface.

    Args:
        x (torch.Tensor): Input tensor.
        weight (torch.Tensor): Weight tensor.
        bias (torch.Tensor): Bias tensor.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        dilation (int): Dilation of the convolution.

    Returns:
        torch.Tensor: Output tensor.
    """
    return F.conv_transpose1d(
        x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation
    )


class Model(nn.Module):
    """
    Performs a transposed 1D convolution operation with square input and asymmetric kernel, optionally with dilation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 3,
        bias: bool = False,
    ):
        super(Model, self).__init__()
        conv = nn.ConvTranspose1d(
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

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.
        """
        return fn(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
        )


def get_inputs(batch_size: int = 16, in_channels: int = 3, length: int = 256):
    x = torch.randn(batch_size, in_channels, length)
    return [x]



input_names = ['x']

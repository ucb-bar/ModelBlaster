import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    dilation: int,
) -> torch.Tensor:
    """
    Performs a standard 1D convolution operation with asymmetric input and a square kernel, potentially dilated and strided.


    Args:
        x (torch.Tensor): Input tensor.
        weight (torch.Tensor): Weight tensor.
        bias (torch.Tensor): Bias tensor.
        stride (int): Stride of the convolution.
        dilation (int): Dilation of the convolution.

    Returns:
        torch.Tensor: Output tensor.
    """
    return F.conv1d(x, weight, bias=bias, stride=stride, dilation=dilation)


class Model(nn.Module):
    """
    Performs a standard 1D convolution operation with asymmetric input and a square kernel, potentially dilated and strided.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int): Stride of the convolution.
        dilation (int): Spacing between kernel elements.
        bias (bool): If `True`, adds a learnable bias to the output.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 3,
        dilation: int = 4,
        bias: bool = False,
    ):
        super(Model, self).__init__()
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

        # Copy the initialized parameters
        self.weight = nn.Parameter(conv.weight.clone())
        self.bias = nn.Parameter(conv.bias.clone()) if bias else None

        self.stride = stride
        self.dilation = dilation

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Performs the 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return fn(x, self.weight, self.bias, self.stride, self.dilation)


def get_inputs(batch_size: int = 16, in_channels: int = 3, length: int = 256):
    x = torch.randn(batch_size, in_channels, length)
    return [x]



input_names = ['x']

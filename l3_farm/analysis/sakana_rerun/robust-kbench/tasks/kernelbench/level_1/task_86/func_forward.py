import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    depthwise_weight: torch.Tensor,
    pointwise_weight: torch.Tensor,
    depthwise_bias: torch.Tensor,
    pointwise_bias: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    """
    Performs a depthwise-separable 2D convolution operation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        depthwise_weight (torch.Tensor): Depthwise convolution weights of shape (in_channels, 1, kernel_size, kernel_size).
        pointwise_weight (torch.Tensor): Pointwise convolution weights of shape (out_channels, in_channels, 1, 1).
        depthwise_bias (torch.Tensor): Depthwise bias of shape (in_channels).
        pointwise_bias (torch.Tensor): Pointwise bias of shape (out_channels).
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        dilation (int): Spacing between kernel elements.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
    """
    x = F.conv2d(
        x,
        depthwise_weight,
        bias=depthwise_bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=depthwise_weight.shape[0],
    )
    x = F.conv2d(x, pointwise_weight, bias=pointwise_bias)
    return x


class Model(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        dilation (int): Spacing between kernel elements.
        bias (bool): If `True`, adds a learnable bias to the output.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super(Model, self).__init__()
        depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.depthwise_weight = nn.Parameter(depthwise.weight.clone())
        self.pointwise_weight = nn.Parameter(pointwise.weight.clone())
        self.depthwise_bias = nn.Parameter(depthwise.bias.clone()) if bias else None
        self.pointwise_bias = nn.Parameter(pointwise.bias.clone()) if bias else None
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
            self.depthwise_weight,
            self.pointwise_weight,
            self.depthwise_bias,
            self.pointwise_bias,
            self.stride,
            self.padding,
            self.dilation,
        )


def get_inputs(
    batch_size: int = 16, in_channels: int = 3, height: int = 256, width: int = 256
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


input_names = ["x"]

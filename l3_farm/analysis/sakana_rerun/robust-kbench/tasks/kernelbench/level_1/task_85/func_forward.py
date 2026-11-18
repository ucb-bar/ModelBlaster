import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride_h: int,
    stride_w: int,
    padding_h: int,
    padding_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
) -> torch.Tensor:
    """
    Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).
        weight (torch.Tensor): Weight tensor of shape (in_channels, out_channels//in_channels, kernel_size_h, kernel_size_w).
        bias (torch.Tensor): Bias tensor of shape (out_channels).
        stride_h (int): Stride of the convolution in height dimension.
        stride_w (int): Stride of the convolution in width dimension.
        padding_h (int): Padding applied to the input in height dimension.
        padding_w (int): Padding applied to the input in width dimension.
        dilation_h (int): Spacing between kernel elements in height dimension.
        dilation_w (int): Spacing between kernel elements in width dimension.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
    """
    return F.conv2d(
        x,
        weight,
        bias=bias,
        stride=(stride_h, stride_w),
        padding=(padding_h, padding_w),
        dilation=(dilation_h, dilation_w),
        groups=groups,
    )


class Model(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel.
    """

    def __init__(
        self,
        in_channels: int = 3,
        kernel_size_h: int = 3,
        kernel_size_w: int = 5,
        stride_h: int = 1,
        stride_w: int = 1,
        padding_h: int = 0,
        padding_w: int = 0,
        dilation_h: int = 1,
        dilation_w: int = 1,
        bias: bool = False,
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(
            in_channels,
            in_channels,
            (kernel_size_h, kernel_size_w),
            stride=(stride_h, stride_w),
            padding=(padding_h, padding_w),
            dilation=(dilation_h, dilation_w),
            groups=in_channels,
        )
        self.weight = nn.Parameter(conv.weight.clone())
        self.bias = nn.Parameter(conv.bias.clone()) if bias else None
        self.stride = (stride_h, stride_w)
        self.padding = (padding_h, padding_w)
        self.dilation = (dilation_h, dilation_w)
        self.groups = in_channels

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.weight,
            self.bias,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
            self.groups,
        )


def get_inputs(
    batch_size: int = 16, in_channels: int = 3, height: int = 128, width: int = 256
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]



input_names = ['x']

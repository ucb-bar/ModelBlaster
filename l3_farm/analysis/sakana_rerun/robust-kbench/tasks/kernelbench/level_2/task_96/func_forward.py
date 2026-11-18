import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    scale: float,
    maxpool_kernel_size: int,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies a transposed 3D convolution, scales the output, applies max pooling,
    global average pooling, and clamps the result.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        stride (int): Stride of the transposed convolution
        padding (int): Padding of the transposed convolution
        scale (float): Scaling factor to multiply output by
        maxpool_kernel_size (int): Kernel size for max pooling operation
        conv_transpose (torch.Tensor): Weight tensor for transposed convolution
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution

    Returns:
        torch.Tensor: Output tensor after applying all operations, with shape
            (batch_size, out_channels, 1, 1, 1)
    """
    x = F.conv_transpose3d(
        x, conv_transpose, bias=conv_transpose_bias, stride=stride, padding=padding
    )
    x = x * scale
    x = F.max_pool3d(x, kernel_size=maxpool_kernel_size)
    x = F.adaptive_avg_pool3d(x, (1, 1, 1))
    x = torch.clamp(x, min=0, max=1)
    return x


class Model(nn.Module):
    """
    Model that performs a transposed 3D convolution, multiplies by a scalar, applies max pooling,
    global average pooling, and clamps the output.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        scale: float = 0.5,
        maxpool_kernel_size: int = 2,
    ):
        super(Model, self).__init__()
        conv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.conv_transpose_parameter = conv.weight
        self.conv_transpose_bias = conv.bias
        self.stride = stride
        self.padding = padding
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.stride,
            self.padding,
            self.scale,
            self.maxpool_kernel_size,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
        )


def get_inputs(
    batch_size: int = 128,
    in_channels: int = 3,
    depth: int = 16,
    height: int = 32,
    width: int = 32,
):
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]



input_names = ['x']

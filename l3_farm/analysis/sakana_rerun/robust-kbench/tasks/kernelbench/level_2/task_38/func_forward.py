import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    output_padding: int,
    pool_kernel_size: int,
    clamp_min: float,
    clamp_max: float,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 3D transposed convolution, average pooling, clamping, softmax and multiplication.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        stride (int): Stride of the transposed convolution
        padding (int): Padding of the transposed convolution
        output_padding (int): Additional size added to output shape
        pool_kernel_size (int): Kernel size for average pooling
        clamp_min (float): Minimum value for clamping
        clamp_max (float): Maximum value for clamping
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution

    Returns:
        torch.Tensor: Output tensor after applying operations
    """
    x = F.conv_transpose3d(
        x,
        conv_transpose,
        bias=conv_transpose_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )
    x = F.avg_pool3d(x, pool_kernel_size)
    x = torch.clamp(x, clamp_min, clamp_max)
    x = F.softmax(x, dim=1)
    x = x * 2
    return x


class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, average pooling, clamping, softmax, and multiplication.
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 16,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
        pool_kernel_size: int = 2,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
    ):
        super(Model, self).__init__()
        conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.conv_transpose_parameter = conv_transpose.weight
        self.conv_transpose_bias = nn.Parameter(
            conv_transpose.bias
            + torch.randn(
                conv_transpose.bias.shape,
                device=conv_transpose.bias.device,
                dtype=conv_transpose.bias.dtype,
            )
            * 0.02
        )
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(
        self,
        x,
        fn=forward_fn,
    ):
        return fn(
            x,
            self.stride,
            self.padding,
            self.output_padding,
            self.pool_kernel_size,
            self.clamp_min,
            self.clamp_max,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
        )


def get_inputs(
    batch_size: int = 16,
    in_channels: int = 8,
    depth: int = 16,
    height: int = 32,
    width: int = 32,
):
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]



input_names = ['x']

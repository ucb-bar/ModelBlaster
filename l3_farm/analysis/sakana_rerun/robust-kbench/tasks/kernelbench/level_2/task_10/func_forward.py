import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    maxpool_kernel_size: int,
    maxpool_stride: int,
    hardtanh_min: float,
    hardtanh_max: float,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies transposed convolution, max pooling, hardtanh, mean and tanh operations.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        stride (int): Stride of the transposed convolution
        padding (int): Padding of the transposed convolution
        maxpool_kernel_size (int): Kernel size for max pooling
        maxpool_stride (int): Stride for max pooling
        hardtanh_min (float): Minimum value for hardtanh
        hardtanh_max (float): Maximum value for hardtanh
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution

    Returns:
        torch.Tensor: Output tensor after applying all operations
    """
    x = F.conv_transpose2d(
        x, conv_transpose, bias=conv_transpose_bias, stride=stride, padding=padding
    )
    x = F.max_pool2d(x, kernel_size=maxpool_kernel_size, stride=maxpool_stride)
    x = F.hardtanh(x, min_val=hardtanh_min, max_val=hardtanh_max)
    x = torch.mean(x, dim=(2, 3), keepdim=True)
    x = torch.tanh(x)
    return x


class Model(nn.Module):
    """
    Model that performs a transposed convolution, followed by max pooling, hardtanh activation, mean operation, and tanh activation.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 64,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        maxpool_kernel_size: int = 2,
        maxpool_stride: int = 2,
        hardtanh_min: float = -1,
        hardtanh_max: float = 1,
    ):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

        self.conv_transpose_weight = self.conv_transpose.weight
        self.conv_transpose_bias = self.conv_transpose.bias
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(
        self,
        x,
        fn=forward_fn,
    ):
        return fn(
            x,
            self.stride,
            self.padding,
            self.maxpool_kernel_size,
            self.maxpool_stride,
            self.hardtanh_min,
            self.hardtanh_max,
            self.conv_transpose_weight,
            self.conv_transpose_bias,
        )


def get_inputs(
    batch_size: int = 128,
    in_channels: int = 32,
    height: int = 16,
    width: int = 16,
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


input_names = ["x"]

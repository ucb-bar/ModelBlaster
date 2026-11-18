import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    conv_transpose_weight: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    sum_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    output_padding: tuple,
    pool_kernel_size: tuple,
    norm_shape: tuple,
) -> torch.Tensor:
    """
    Functional implementation of a sequence of operations:
    1. 3D transposed convolution
    2. Addition with a learnable weight
    3. Layer normalization
    4. 3D average pooling
    5. GELU activation

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        conv_transpose_weight (torch.Tensor): Weight tensor for transposed convolution
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        sum_weight (torch.Tensor): Learnable weight for addition
        norm_weight (torch.Tensor): Weight tensor for layer normalization
        norm_bias (torch.Tensor): Bias tensor for layer normalization
        stride (tuple): Stride for transposed convolution, as (depth_stride, height_stride, width_stride)
        padding (tuple): Padding for transposed convolution, as (depth_pad, height_pad, width_pad)
        output_padding (tuple): Output padding for transposed convolution, as (depth_pad, height_pad, width_pad)
        pool_kernel_size (tuple): Kernel size for average pooling, as (depth_kernel, height_kernel, width_kernel)
        norm_shape (tuple): Shape for layer normalization

    Returns:
        torch.Tensor: Output tensor after applying all operations
    """
    x = F.conv_transpose3d(
        x,
        conv_transpose_weight,
        bias=conv_transpose_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )
    x = x + sum_weight
    x = F.layer_norm(x, norm_shape, norm_weight, norm_bias)
    x = F.avg_pool3d(x, kernel_size=pool_kernel_size)
    x = F.gelu(x)
    return x


class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum, layer normalization, average pooling, and GELU activation.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 64,
        kernel_size: tuple = (3, 3, 3),
        stride: tuple = (2, 2, 2),
        padding: tuple = (1, 1, 1),
        output_padding: tuple = (1, 1, 1),
        sum_weight: float = 1.0,
        pool_kernel_size: tuple = (2, 2, 2),
    ):
        super(Model, self).__init__()
        conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.conv_transpose_weight = nn.Parameter(conv.weight)
        self.conv_transpose_bias = nn.Parameter(conv.bias)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm_shape = (out_channels,)
        norm = nn.LayerNorm(self.norm_shape)
        self.norm_weight = nn.Parameter(
            norm.weight + torch.randn(self.norm_shape) * 0.02
        )
        self.norm_bias = nn.Parameter(norm.bias + torch.randn(self.norm_shape) * 0.02)
        self.pool_kernel_size = pool_kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(
        self,
        x,
        fn=forward_fn,
    ):
        return fn(
            x,
            self.conv_transpose_weight,
            self.conv_transpose_bias,
            self.sum_weight,
            self.norm_weight,
            self.norm_bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.pool_kernel_size,
            self.norm_shape,
        )


def get_inputs(
    batch_size: int = 128,
    in_channels: int = 32,
    depth: int = 16,
    height: int = 32,
    width: int = 32,
):
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


input_names = ["x"]

import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    output_padding: int,
    scaling_factor: float,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Applies transposed convolution, bias addition, clamping, scaling, clamping and division.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        stride (int): Stride of the convolution
        padding (int): Zero-padding added to both sides of input
        output_padding (int): Additional size added to output shape
        scaling_factor (float): Factor to scale the tensor by
        conv_transpose (torch.Tensor): Transposed convolution weights
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        bias (torch.Tensor): Bias tensor to add after convolution

    Returns:
        torch.Tensor: Output tensor after applying operations
    """
    x = F.conv_transpose2d(
        x,
        conv_transpose,
        bias=conv_transpose_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )
    x = x + bias
    x = torch.clamp(x, min=0.0, max=1.0)
    x = x * scaling_factor
    x = torch.clamp(x, min=0.0, max=1.0)
    x = x / scaling_factor
    return x


class Model(nn.Module):
    """
    Model that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
        scaling_factor: float = 2.0,
    ):
        super(Model, self).__init__()
        conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            output_padding=output_padding,
        )
        bias_shape = (out_channels, 1, 1)
        self.conv_transpose_parameter = nn.Parameter(conv_transpose.weight)
        self.conv_tranpose_bias = nn.Parameter(conv_transpose.bias)
        self.bias_parameter = nn.Parameter(torch.randn(bias_shape) * 0.02)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scaling_factor = scaling_factor

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.stride,
            self.padding,
            self.output_padding,
            self.scaling_factor,
            self.conv_transpose_parameter,
            self.conv_tranpose_bias,
            self.bias_parameter,
        )


def get_inputs(
    batch_size: int = 128, in_channels: int = 3, height: int = 32, width: int = 32
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


input_names = ["x"]

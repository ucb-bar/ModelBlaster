import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    output_padding: int,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    add_value: float,
    scale: float,
) -> torch.Tensor:
    """
    Applies transposed convolution, Mish activation, adds a value, applies Hardtanh, and scales.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        add_value (float): Value to add after Mish activation
        scale (float): Value to multiply output by

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
    x = F.mish(x)
    x = x + add_value
    x = F.hardtanh(x, min_val=-1, max_val=1)
    x = x * scale
    return x


class Model(nn.Module):
    """
    Model that performs a transposed convolution, applies Mish activation, adds a value,
    applies Hardtanh activation, and scales the output.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 64,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
        add_value: float = 0.5,
        scale: float = 2,
    ):
        super(Model, self).__init__()
        conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.conv_transpose_weight = conv_transpose.weight
        self.conv_transpose_bias = conv_transpose.bias
        self.add_value = add_value
        self.scale = scale
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.stride,
            self.padding,
            self.output_padding,
            self.conv_transpose_weight,
            self.conv_transpose_bias,
            self.add_value,
            self.scale,
        )


def get_inputs(
    batch_size: int = 128, in_channels: int = 32, height: int = 16, width: int = 16
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


input_names = ["x"]

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
    bias: torch.Tensor,
    scaling_factor: float,
) -> torch.Tensor:
    """
    Applies transposed convolution, softmax, bias addition, scaling and sigmoid.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        bias (torch.Tensor): Bias tensor for addition
        scaling_factor (float): Factor to scale the output by

    Returns:
        torch.Tensor: Output tensor after applying all operations
    """
    x = F.conv_transpose2d(
        x,
        conv_transpose,
        bias=conv_transpose_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )
    x = F.softmax(x, dim=1)
    x = x + bias
    x = x * scaling_factor
    x = torch.sigmoid(x)
    return x


class Model(nn.Module):
    """
    Model that performs a transposed convolution, applies softmax, adds a bias term,
    scales the result, and applies sigmoid.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 64,
        kernel_size: int = 4,
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
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.conv_transpose_parameter = conv_transpose.weight
        self.conv_transpose_bias = conv_transpose.bias
        bias_shape = (out_channels, 1, 1)
        self.bias_parameter = nn.Parameter(torch.randn(bias_shape) * 0.02)
        self.scaling_factor = scaling_factor
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.stride,
            self.padding,
            self.output_padding,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
            self.bias_parameter,
            self.scaling_factor,
        )


def get_inputs(
    batch_size: int = 128,
    in_channels: int = 32,
    out_channels: int = 64,
    height: int = 16,
    width: int = 16,
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]



input_names = ['x']

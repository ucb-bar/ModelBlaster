import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    bias: torch.Tensor,
    scaling_factor: float,
    stride: int,
    padding: int,
) -> torch.Tensor:
    """
    Applies a series of operations:
    1. Transposed 3D convolution
    2. Mean pooling
    3. Addition
    4. Softmax
    5. Tanh activation
    6. Scaling

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        bias (torch.Tensor): Bias tensor for addition
        scaling_factor (float): Scaling factor for final multiplication
        stride (int): Stride for transposed convolution
        padding (int): Padding for transposed convolution

    Returns:
        torch.Tensor: Output tensor after applying all operations
    """
    x = F.conv_transpose3d(
        x, conv_transpose, bias=conv_transpose_bias, stride=stride, padding=padding
    )
    x = torch.mean(x, dim=1, keepdim=True)
    x = x + bias
    x = F.softmax(x, dim=1)
    x = torch.tanh(x)
    x = x * scaling_factor
    return x


class Model(nn.Module):
    """
    Model that performs a series of operations:
    1. Transposed 3D convolution
    2. Mean pooling
    3. Addition
    4. Softmax
    5. Tanh activation
    6. Scaling
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 16,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        scaling_factor: float = 2.0,
    ):
        super(Model, self).__init__()
        conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        bias_shape = (1, 1, 1, 1, 1)
        self.bias = nn.Parameter(torch.randn(bias_shape) * 0.02)

        self.conv_transpose_weight = conv_transpose.weight
        self.conv_transpose_bias = conv_transpose.bias
        self.scaling_factor = scaling_factor
        self.stride = stride
        self.padding = padding

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.conv_transpose_weight,
            self.conv_transpose_bias,
            self.bias,
            self.scaling_factor,
            self.stride,
            self.padding,
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

import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    scale1: torch.Tensor,
    scale2: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies a 3D transposed convolution, scaling, average pooling, bias addition and scaling.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        stride (int): Stride of the transposed convolution
        padding (int): Padding of the transposed convolution
        conv_transpose (torch.Tensor): Transposed convolution weight tensor
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        scale1 (torch.Tensor): First scaling factor
        scale2 (torch.Tensor): Second scaling factor
        bias (torch.Tensor): Bias tensor for addition

    Returns:
        torch.Tensor: Output tensor after applying operations
    """
    x = F.conv_transpose3d(
        x, conv_transpose, bias=conv_transpose_bias, stride=stride, padding=padding
    )
    x = x * scale1
    x = F.avg_pool3d(x, kernel_size=2)
    x = x + bias
    x = x * scale2
    return x


class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, scaling, average pooling, bias addition, and scaling.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        scale1: float = 0.5,
        scale2: float = 1.0,
    ):
        super(Model, self).__init__()
        conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.conv_transpose_parameter = nn.Parameter(conv_transpose.weight)
        self.conv_transpose_bias = nn.Parameter(
            conv_transpose.bias
            + torch.randn(
                conv_transpose.bias.shape,
                device=conv_transpose.bias.device,
                dtype=conv_transpose.bias.dtype,
            )
            * 0.02
        )
        self.scale1_parameter = nn.Parameter(torch.tensor(scale1))
        self.scale2_parameter = nn.Parameter(torch.tensor(scale2))
        bias_shape = (out_channels, 1, 1, 1)
        self.bias_parameter = nn.Parameter(torch.randn(bias_shape) * 0.02)
        self.stride = stride
        self.padding = padding

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.stride,
            self.padding,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
            self.scale1_parameter,
            self.scale2_parameter,
            self.bias_parameter,
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

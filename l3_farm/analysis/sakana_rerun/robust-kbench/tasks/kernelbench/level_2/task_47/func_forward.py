import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    stride: int,
    padding: int,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 3D convolution followed by Mish and Tanh activations.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        stride (int): Stride of the convolution
        padding (int): Padding of the convolution
        conv_weight (torch.Tensor): Convolution weight tensor of shape
            (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        conv_bias (torch.Tensor): Bias tensor for convolution of shape (out_channels)

    Returns:
        torch.Tensor: Output tensor after applying convolution, Mish and Tanh activations
    """
    x = F.conv3d(x, conv_weight, bias=conv_bias, stride=stride, padding=padding)
    x = F.mish(x)
    x = torch.tanh(x)
    return x


class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Mish activation, and then applies Tanh activation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ):
        super(Model, self).__init__()
        conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(
            conv.bias
            + torch.randn(
                conv.bias.shape, device=conv.bias.device, dtype=conv.bias.dtype
            )
            * 0.02
        )
        self.stride = stride
        self.padding = padding

    def forward(self, x, fn=forward_fn):
        return fn(x, self.stride, self.padding, self.conv_weight, self.conv_bias)


def get_inputs(
    batch_size: int = 16, in_channels: int = 3, D: int = 16, H: int = 32, W: int = 32
):
    x = torch.randn(batch_size, in_channels, D, H, W)
    return [x]



input_names = ['x']

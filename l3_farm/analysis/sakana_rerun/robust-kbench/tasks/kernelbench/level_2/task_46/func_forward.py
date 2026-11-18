import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    kernel_size_pool: int,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    subtract1_value: float,
    subtract2_value: float,
) -> torch.Tensor:
    """
    Applies convolution, subtraction, tanh activation, subtraction and average pooling.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        kernel_size_pool (int): Kernel size for average pooling
        conv_weight (torch.Tensor): Convolution weight tensor
        conv_bias (torch.Tensor): Convolution bias tensor
        subtract1_value (float): First subtraction value
        subtract2_value (float): Second subtraction value

    Returns:
        torch.Tensor: Output tensor after applying convolution, subtractions, tanh and avg pooling
    """
    x = F.conv2d(x, conv_weight, bias=conv_bias)
    x = x - subtract1_value
    x = torch.tanh(x)
    x = x - subtract2_value
    x = F.avg_pool2d(x, kernel_size_pool)
    return x


class Model(nn.Module):
    """
    Model that performs a convolution, subtraction, tanh activation, subtraction and average pooling.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        subtract1_value: float = 0.5,
        subtract2_value: float = 0.2,
        kernel_size_pool: int = 2,
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(
            conv.bias
            + torch.randn(
                conv.bias.shape, device=conv.bias.device, dtype=conv.bias.dtype
            )
            * 0.02
        )
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.kernel_size_pool,
            self.conv_weight,
            self.conv_bias,
            self.subtract1_value,
            self.subtract2_value,
        )


def get_inputs(
    batch_size: int = 128, in_channels: int = 3, height: int = 32, width: int = 32
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]



input_names = ['x']

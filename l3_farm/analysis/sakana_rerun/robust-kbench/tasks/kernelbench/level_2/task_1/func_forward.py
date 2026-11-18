import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Functional implementation of a neural network layer that:
    1. Applies a 2D convolution with learnable weights and biases
    2. Applies ReLU activation function
    3. Adds a learnable bias term

    Args:
        x (Tensor): Input tensor of shape (N, C_in, H, W)
        conv_weight (Tensor): Convolution weights of shape (C_out, C_in, kernel_size, kernel_size)
        conv_bias (Tensor): Convolution bias of shape (C_out)
        bias (Tensor): Additional bias term of shape (C_out, 1, 1)

    Returns:
        Tensor: Output tensor of shape (N, C_out, H_out, W_out)
    """
    x = F.conv2d(x, conv_weight, conv_bias)
    x = torch.relu(x)
    x = x + bias
    return x


class Model(nn.Module):
    """
    Simple model that performs a convolution, applies ReLU, and adds a bias term.
    """

    def __init__(
        self, in_channels: int = 3, out_channels: int = 16, kernel_size: int = 3
    ):
        super(Model, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv_weight = nn.Parameter(conv.weight)
        self.conv_bias = nn.Parameter(conv.bias)
        self.bias = nn.Parameter(torch.randn((out_channels, 1, 1)) * 0.02)

    def forward(self, x, fn=forward_fn):
        return fn(x, self.conv_weight, self.conv_bias, self.bias)


def get_inputs(
    batch_size: int = 128, in_channels: int = 3, height: int = 32, width: int = 32
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


input_names = ["x"]

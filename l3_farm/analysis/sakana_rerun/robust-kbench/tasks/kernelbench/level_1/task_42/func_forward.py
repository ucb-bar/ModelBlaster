import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    """
    Applies Max Pooling 2D using functional interface.

    Args:
        x (torch.Tensor): Input tensor
        kernel_size (int): Size of pooling window
        stride (int): Stride of pooling window
        padding (int): Padding to be applied
        dilation (int): Spacing between kernel elements

    Returns:
        torch.Tensor: Output tensor after max pooling
    """
    return F.max_pool2d(
        x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )


class Model(nn.Module):
    """
    Simple model that performs Max Pooling 2D.
    """

    def __init__(
        self, kernel_size: int = 2, stride: int = 2, padding: int = 1, dilation: int = 3
    ):
        """
        Initializes the model parameters.
        """
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Forward pass that calls forward_fn.
        """
        return fn(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )


def get_inputs(
    batch_size: int = 16, channels: int = 32, height: int = 128, width: int = 128
):
    x = torch.randn(batch_size, channels, height, width)
    return [x]



input_names = ['x']

import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor, kernel_size: int, stride: int, padding: int
) -> torch.Tensor:
    """
    Applies 1D Average Pooling using functional interface.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length)
        kernel_size (int): Size of the pooling window
        stride (int): Stride of the pooling operation
        padding (int): Padding applied to the input tensor

    Returns:
        torch.Tensor: Output tensor with 1D Average Pooling applied
    """
    return F.avg_pool1d(x, kernel_size=kernel_size, stride=stride, padding=padding)


class Model(nn.Module):
    """
    Simple model that performs 1D Average Pooling.
    """

    def __init__(self, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        """
        Initializes the 1D Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window
            stride (int): Stride of the pooling operation
            padding (int): Padding applied to the input tensor
        """
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Applies 1D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length)
            fn: Function to apply pooling operation, defaults to forward_fn

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied
        """
        return fn(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
        )


def get_inputs(batch_size: int = 16, in_channels: int = 32, input_length: int = 128):
    x = torch.randn(batch_size, in_channels, input_length)
    return [x]



input_names = ['x']

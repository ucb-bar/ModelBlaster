import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor, kernel_size: int, stride: int, padding: int
) -> torch.Tensor:
    """
    Applies 2D Average Pooling using functional interface.

    Args:
        x (torch.Tensor): Input tensor
        kernel_size (int): Size of pooling window
        stride (int): Stride of pooling operation
        padding (int): Input padding

    Returns:
        torch.Tensor: Output tensor with 2D Average Pooling applied
    """
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)


class Model(nn.Module):
    """
    Simple model that performs 2D Average Pooling.
    """

    def __init__(self, kernel_size: int = 3, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window
            stride (int): Stride of the pooling operation
            padding (int): Padding applied to input tensor
        """
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            fn: Function to apply pooling operation, defaults to forward_fn

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied
        """
        return fn(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
        )


def get_inputs(
    batch_size: int = 16, channels: int = 64, height: int = 256, width: int = 256
):
    x = torch.randn(batch_size, channels, height, width)
    return [x]



input_names = ['x']

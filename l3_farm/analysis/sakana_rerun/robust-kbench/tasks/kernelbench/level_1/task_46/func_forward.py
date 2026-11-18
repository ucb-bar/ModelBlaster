import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor, kernel_size: int, stride: int, padding: int
) -> torch.Tensor:
    """
    Applies 3D Average Pooling using functional interface.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width)
        kernel_size (int): Size of the kernel to apply pooling
        stride (int): Stride of the pooling operation
        padding (int): Padding to apply before pooling

    Returns:
        torch.Tensor: Output tensor with Average Pooling applied
    """
    return F.avg_pool3d(x, kernel_size=kernel_size, stride=stride, padding=padding)


class Model(nn.Module):
    """
    Simple model that performs 3D Average Pooling.
    """

    def __init__(self, kernel_size: int = 3, stride: int = 2, padding: int = 1):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int): Stride of the pooling operation.
            padding (int): Padding to apply before pooling.
        """
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).
            fn: Function to apply pooling operation. Defaults to forward_fn.

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied, shape depends on kernel_size, stride and padding.
        """
        return fn(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
        )


def get_inputs(
    batch_size: int = 16,
    channels: int = 32,
    depth: int = 64,
    height: int = 64,
    width: int = 64,
):
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]


input_names = ["x"]

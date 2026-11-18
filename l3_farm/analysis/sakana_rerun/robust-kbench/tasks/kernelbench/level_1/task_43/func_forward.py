import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    return_indices: bool,
    ceil_mode: bool,
) -> torch.Tensor:
    """
    Functional implementation of Max Pooling 3D.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).
        kernel_size (int): Size of the kernel for the max pooling operation.
        stride (int): Stride of the pooling operation.
        padding (int): Padding applied to the input tensor.
        dilation (int): Spacing between kernel elements.
        return_indices (bool): Whether to return indices of the maximum values.
        ceil_mode (bool): When True, the output size is ceil(input_size / stride) instead of floor.

    Returns:
        torch.Tensor: Output tensor with Max Pooling 3D applied.
    """
    return F.max_pool3d(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=return_indices,
        ceil_mode=ceil_mode,
    )


class Model(nn.Module):
    """
    Simple model that performs Max Pooling 3D.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        dilation: int = 3,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        """
        Initializes the Max Pooling 3D layer.

        Args:
            kernel_size (int): Size of the kernel for the max pooling operation.
            stride (int): Stride of the pooling operation.
            padding (int): Padding applied to the input tensor.
            dilation (int): Spacing between kernel elements.
            return_indices (bool): Whether to return indices of the maximum values.
            ceil_mode (bool): When True, the output size is ceil(input_size / stride) instead of floor.
        """
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 3D applied.
        """
        return fn(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices,
            self.ceil_mode,
        )


def get_inputs(
    batch_size: int = 16,
    channels: int = 32,
    dim1: int = 64,
    dim2: int = 64,
    dim3: int = 64,
):
    x = torch.randn(batch_size, channels, dim1, dim2, dim3)
    return [x]



input_names = ['x']

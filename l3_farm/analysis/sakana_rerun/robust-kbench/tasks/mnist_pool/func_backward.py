import torch
import torch.nn as nn
import torch.nn.functional as F


class AutogradFunction(torch.autograd.Function):
    backward_fn = None

    @staticmethod
    def forward(ctx, x):
        # Save inputs for backward pass
        ctx.save_for_backward(x)
        # Apply max pooling
        x = F.max_pool2d(x, kernel_size=2)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved inputs
        x = ctx.saved_tensors

        # Use the class-level backward function
        grad_input = AutogradFunction.backward_fn(x, grad_output)
        return grad_input


def forward_fn(
    x: torch.Tensor,
) -> torch.Tensor:
    """Implements a max pooling layer with kernel size 2:

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        kernel_size (int): Kernel size of the max pooling layer

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width)
    """
    # Apply max pooling
    x = F.max_pool2d(x, kernel_size=2)
    return x


class Model(nn.Module):
    """
    Simple model that performs a max pooling layer with kernel size 2.
    """

    def __init__(
        self,
        kernel_size: int = 2,
    ):
        """
        Initializes the max pooling layer with kernel size 2.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Forward pass that calls forward_fn.
        """
        return fn(x)


def get_inputs(
    batch_size: int = 64,
    in_channels: int = 64,
    height: int = 28,
    width: int = 28,
):
    x = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
    return [x]


input_names = ["x"]

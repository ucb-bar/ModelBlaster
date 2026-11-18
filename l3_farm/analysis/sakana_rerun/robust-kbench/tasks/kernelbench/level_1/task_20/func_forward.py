import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(x: torch.Tensor, negative_slope: float) -> torch.Tensor:
    """
    Applies LeakyReLU activation to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of any shape.
        negative_slope (float): The negative slope of the activation function.

    Returns:
        torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
    """
    return F.leaky_relu(x, negative_slope)


class Model(nn.Module):
    """
    Simple model that performs a LeakyReLU activation.
    """

    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float): The negative slope of the activation function.
        """
        super(Model, self).__init__()
        self.negative_slope_param = negative_slope

    def forward(self, x, fn=forward_fn):
        """
        Applies LeakyReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            fn (callable): Function to compute the forward pass. Defaults to forward_fn.

        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        return fn(x, self.negative_slope_param)


def get_inputs(batch_size: int = 16, dim: int = 16384):
    x = torch.randn(batch_size, dim)
    return [x]


input_names = ['x']

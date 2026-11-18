import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Applies ELU activation to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of any shape.
        alpha (float): The alpha parameter for the ELU function.

    Returns:
        torch.Tensor: Output tensor with ELU applied, same shape as input.
    """
    return F.elu(x, alpha=alpha)


class Model(nn.Module):
    """
    Simple model that performs an ELU activation.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initializes the ELU model.

        Args:
            alpha (float): The alpha parameter for the ELU function.
        """
        super(Model, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Applies ELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ELU applied, same shape as input.
        """
        return fn(x, self.alpha)


def get_inputs(batch_size: int = 16, dim: int = 16384):
    x = torch.randn(batch_size, dim)
    return [x]



input_names = ['x']

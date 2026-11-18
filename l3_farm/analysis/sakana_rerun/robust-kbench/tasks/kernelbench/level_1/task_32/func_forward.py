import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    Applies HardTanh activation to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of any shape.
        min_val (float): The minimum value for the HardTanh function.
        max_val (float): The maximum value for the HardTanh function.

    Returns:
        torch.Tensor: Output tensor with HardTanh applied, same shape as input.
    """
    return F.hardtanh(x, min_val=min_val, max_val=max_val)


class Model(nn.Module):
    """
    Simple model that performs a HardTanh activation.
    """

    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        super(Model, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Applies HardTanh activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardTanh applied, same shape as input.
        """
        return fn(x, self.min_val, self.max_val)


def get_inputs(batch_size: int = 16, dim: int = 16384):
    x = torch.randn(batch_size, dim)
    return [x]


input_names = ["x"]

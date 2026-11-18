import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Simple model that performs a HardTanh activation.
    """

    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        super(Model, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardTanh activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardTanh applied, same shape as input.
        """
        return F.hardtanh(x, min_val=self.min_val, max_val=self.max_val)


batch_size = 16
dim = 16384
min_val = -1.0
max_val = 1.0


def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]


def get_init_inputs():
    return [min_val, max_val]

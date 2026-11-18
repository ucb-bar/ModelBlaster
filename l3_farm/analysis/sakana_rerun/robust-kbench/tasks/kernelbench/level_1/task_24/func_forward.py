import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Applies LogSoftmax activation to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim)
        dim (int): Dimension along which to apply LogSoftmax

    Returns:
        torch.Tensor: Output tensor with LogSoftmax applied, same shape as input
    """
    return F.log_softmax(x, dim=dim)


class Model(nn.Module):
    """
    Simple model that performs a LogSoftmax activation.
    """

    def __init__(self, sm_dim: int = 1):
        super(Model, self).__init__()
        self.sm_dim = sm_dim

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        return fn(x, self.sm_dim)


def get_inputs(batch_size: int = 16, dim: int = 16384):
    x = torch.randn(batch_size, dim)
    return [x]


input_names = ['x']

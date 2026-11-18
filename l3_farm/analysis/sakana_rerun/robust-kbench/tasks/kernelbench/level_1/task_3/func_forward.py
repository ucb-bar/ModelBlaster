import torch
import torch.nn as nn


def forward_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.

    Args:
        A: Input tensor of shape (batch_size, m, k).
        B: Input tensor of shape (batch_size, k, n).

    Returns:
        C: Output tensor of shape (batch_size, m, n).
    """
    return torch.bmm(A, B)


class Model(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        return fn(A, B)


def get_inputs(batch_size: int = 128, m: int = 128, k: int = 256, n: int = 512):
    A = torch.randn(batch_size, m, k)
    B = torch.randn(batch_size, k, n)
    return [A, B]


input_names = ["A", "B"]

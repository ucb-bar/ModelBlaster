import torch
import torch.nn as nn


def forward_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs a single matrix multiplication (C = A * B) with a small K dimension.

    Args:
        A: Input tensor of shape (M, K).
        B: Input tensor of shape (K, N).

    Returns:
        Output tensor of shape (M, N).
    """
    return torch.matmul(A, B)


class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a small K dimension
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        return fn(A, B)


def get_inputs(M: int = 16384, N: int = 16384, K: int = 32):
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]



input_names = ['A', 'B']

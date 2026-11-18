import torch
import torch.nn as nn


def forward_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs a single matrix multiplication (C = A * B) with irregular shapes.

    Args:
        A: Input tensor with shape (M, K).
        B: Input tensor with shape (K, N).

    Returns:
        C: Output tensor with shape (M, N).
    """
    return torch.matmul(A, B)


class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with irregular shapes
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        return fn(A, B)


def get_inputs(M: int = 8205, K: int = 2949, N: int = 5921):
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]



input_names = ['A', 'B']

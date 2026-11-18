import torch
import torch.nn as nn


def forward_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix-vector multiplication (C = A * B).

    Args:
        A: Input matrix of shape (M, K).
        B: Input vector of shape (K, 1).

    Returns:
        Output vector of shape (M, 1).
    """
    return torch.matmul(A, B)


class Model(nn.Module):
    """
    Simple model that performs matrix-vector multiplication (C = A * B).
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        return fn(A, B)


def get_inputs(M: int = 256, K: int = 131072):
    A = torch.randn(M, K)
    B = torch.randn(K, 1)
    return [A, B]



input_names = ['A', 'B']

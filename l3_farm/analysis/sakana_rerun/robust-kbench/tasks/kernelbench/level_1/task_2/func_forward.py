import torch
import torch.nn as nn


def forward_fn(A, B):
    """
    performs a single general matrix multiplication (C = A * B).

    Args:
        A: Input tensor of shape (M, K).
        B: Input tensor of shape (K, N).

    Returns:
        Output tensor of shape (M, N).
    """
    return torch.matmul(A, B)


class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        return fn(A, B)


def get_inputs(M: int = 1024, K: int = 4096, N: int = 2048):
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]


input_names = ["A", "B"]

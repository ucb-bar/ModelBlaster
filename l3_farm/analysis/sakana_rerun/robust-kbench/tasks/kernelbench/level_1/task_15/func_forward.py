import torch
import torch.nn as nn


def forward_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs a matrix multiplication (C = A * B) where A and B are lower triangular matrices.

    Args:
        A (torch.Tensor): Lower triangular matrix of shape (N, N).
        B (torch.Tensor): Lower triangular matrix of shape (N, N).

    Returns:
        torch.Tensor: The result of matrix multiplication C of shape (N, N).
    """
    return torch.tril(torch.matmul(A, B))


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication (C = A * B) where A and B are lower triangular matrices.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        return fn(A, B)


def get_inputs(M: int = 4096):
    A = torch.randn(M, M)
    B = torch.randn(M, M)
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]



input_names = ["A", "B"]

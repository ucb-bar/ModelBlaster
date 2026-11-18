import torch
import torch.nn as nn


def forward_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs a matrix multiplication of a diagonal matrix with another matrix.

    Args:
        A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
        B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

    Returns:
        torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
    """
    return torch.diag(A) @ B


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        return fn(A, B)


def get_inputs(M: int = 4096, N: int = 4096):
    A = torch.randn(N)
    B = torch.randn(N, M)
    return [A, B]



input_names = ["A", "B"]

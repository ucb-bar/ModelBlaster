import torch
import torch.nn as nn


def forward_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication (C = A * B) for upper triangular matrices.

    Args:
        A (torch.Tensor): Upper triangular matrix of shape (N, N).
        B (torch.Tensor): Upper triangular matrix of shape (N, N).

    Returns:
        torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
    """
    return torch.triu(torch.matmul(A, B))


class Model(nn.Module):
    """
    Simple model that performs matrix multiplication (C = A * B) for upper triangular matrices.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Performs matrix multiplication for upper triangular matrices.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return fn(A, B)


def get_inputs(N: int = 4096):
    """
    Generates upper triangular matrices for testing.

    Returns:
        list: A list containing two upper triangular matrices of shape (N, N).
    """
    A = torch.triu(torch.randn(N, N))
    B = torch.triu(torch.randn(N, N))
    return [A, B]


input_names = ["A", "B"]

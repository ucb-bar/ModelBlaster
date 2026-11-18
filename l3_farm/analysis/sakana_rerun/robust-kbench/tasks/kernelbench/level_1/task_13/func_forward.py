import torch
import torch.nn as nn


def forward_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs a single matrix multiplication (C = A * B) with A and B being symmetric matrices.

    Args:
        A (torch.Tensor): Input matrix A, shape (N, N), symmetric.
        B (torch.Tensor): Input matrix B, shape (N, N), symmetric.

    Returns:
        torch.Tensor: Output matrix C, shape (N, N).
    """
    return torch.matmul(A, B)


class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with A and B being symmetric matrices.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Performs matrix multiplication of two symmetric matrices.

        Args:
            A (torch.Tensor): Input matrix A, shape (N, N), symmetric.
            B (torch.Tensor): Input matrix B, shape (N, N), symmetric.

        Returns:
            torch.Tensor: Output matrix C, shape (N, N).
        """
        return fn(A, B)


def get_inputs(N: int = 4096):
    """
    Generates a pair of random symmetric matrices for testing.

    Returns:
        list: List containing two symmetric tensors A and B.
    """
    A = torch.randn(N, N)
    A = (A + A.T) / 2  # Ensure symmetry
    B = torch.randn(N, N)
    B = (B + B.T) / 2  # Ensure symmetry
    return [A, B]


input_names = ["A", "B"]

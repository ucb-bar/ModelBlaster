import torch
import torch.nn as nn


def forward_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs a single square matrix multiplication (C = A * B).

    Args:
        A (torch.Tensor): Input matrix A of shape (N, N).
        B (torch.Tensor): Input matrix B of shape (N, N).

    Returns:
        torch.Tensor: Output matrix C of shape (N, N).
    """
    return torch.matmul(A, B)


class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        return fn(A, B)


def get_inputs(N: int = 2048):
    A = torch.randn(N, N)
    B = torch.randn(N, N)
    return [A, B]


input_names = ["A", "B"]

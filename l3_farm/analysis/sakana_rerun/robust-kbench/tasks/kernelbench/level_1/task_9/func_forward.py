import torch
import torch.nn as nn


def forward_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M).

    Args:
        A (torch.Tensor): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
        B (torch.Tensor): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.

    Returns:
        torch.Tensor: Output matrix of shape (M, N) or (N, M)
    """
    return torch.matmul(A, B)


class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        return fn(A, B)


def get_inputs(M: int = 16384, N: int = 16):
    A = torch.randn(M, N)
    B = torch.randn(N, M)
    return [A, B]



input_names = ['A', 'B']

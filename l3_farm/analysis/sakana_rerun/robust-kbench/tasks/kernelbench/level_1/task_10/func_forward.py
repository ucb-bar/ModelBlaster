import torch
import torch.nn as nn


def forward_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs 3D tensor-matrix multiplication.

    Args:
        A (torch.Tensor): Input 3D tensor of shape (N, M, K).
        B (torch.Tensor): Input matrix of shape (K, L).

    Returns:
        torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
    """
    return torch.matmul(A, B)


class Model(nn.Module):
    """
    Performs 3D tensor-matrix multiplication.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        return fn(A, B)


def get_inputs(N: int = 16, M: int = 1024, K: int = 2048, L: int = 768):
    A = torch.randn(N, M, K)
    B = torch.randn(K, L)
    return [A, B]



input_names = ['A', 'B']

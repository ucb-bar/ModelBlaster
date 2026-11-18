import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(A, B):
    """
    Performs a single matrix multiplication (C = A * B) with a large K dimension.

    Args:
        A: Input tensor of shape (M, K)
        B: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N)
    """
    return torch.matmul(A, B)


class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a large K dimension
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        return fn(A, B)


def get_inputs(M: int = 256, N: int = 256, K: int = 131072):
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]



input_names = ['A', 'B']

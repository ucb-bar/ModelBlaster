import torch
import torch.nn as nn


def forward_fn(A: torch.Tensor, s: float) -> torch.Tensor:
    """
    Performs a matrix-scalar multiplication (C = A * s).

    Args:
        A: Input matrix of shape (M, N)
        s: Scalar value

    Returns:
        C: Resulting matrix of shape (M, N)
    """
    return A * s


class Model(nn.Module):
    """
    Simple model that performs a matrix-scalar multiplication (C = A * s)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, s: float, fn=forward_fn) -> torch.Tensor:
        return fn(A, s)


def get_inputs(M: int = 16384, N: int = 4096, s: float = 3.14):
    A = torch.randn(M, N)
    return [A, s]



input_names = ['A', 's']

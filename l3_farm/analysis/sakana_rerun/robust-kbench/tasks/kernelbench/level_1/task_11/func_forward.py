import torch
import torch.nn as nn


def forward_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs 4D tensor-matrix multiplication:
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)

    Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """
    return torch.einsum("bijl,lk->bijk", A, B)


class Model(nn.Module):
    """
    Performs 4D tensor-matrix multiplication:
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)

    Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        return fn(A, B)


def get_inputs(b: int = 16, i: int = 16, j: int = 16, l: int = 16, k: int = 16):
    A = torch.randn(b, i, j, l)
    B = torch.randn(l, k)
    return [A, B]



input_names = ["A", "B"]

import torch
import torch.nn as nn
import math


def forward_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the Gaussian Error Linear Units (GELU) activation function currently in Google BERT repo (identical to OpenAI GPT).

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor.
    """
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )


class Model(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, fn=forward_fn):
        return fn(x)


def get_inputs(batch_size: int = 2000, dim: int = 2000):
    x = torch.randn(batch_size, dim)
    return [x]


input_names = ["x"]

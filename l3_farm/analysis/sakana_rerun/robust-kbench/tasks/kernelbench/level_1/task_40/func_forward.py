import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """
    Functional implementation of LayerNorm.

    Args:
        x (torch.Tensor): Input tensor of shape (*, normalized_shape).
        weight (torch.Tensor): Weight tensor of shape (normalized_shape).
        bias (torch.Tensor): Bias tensor of shape (normalized_shape).
        eps (float): Epsilon parameter for numerical stability.

    Returns:
        torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
    """
    # Get the normalized shape from the weight tensor
    normalized_shape = tuple(x.shape[-len(weight.shape) :])
    return F.layer_norm(
        x, normalized_shape=normalized_shape, weight=weight, bias=bias, eps=eps
    )


class Model(nn.Module):
    """
    Simple model that performs Layer Normalization.
    """

    def __init__(self, num_features: int = 64, dim1: int = 256, dim2: int = 256):
        """
        Initializes the LayerNorm layer parameters.

        Args:
            num_features (int): Number of features in the input tensor.
            dim1 (int): First dimension of the input tensor.
            dim2 (int): Second dimension of the input tensor.
        """
        super(Model, self).__init__()
        normalized_shape = (num_features, dim1, dim2)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).
            fn: Function to apply (defaults to forward_fn)

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return fn(x, self.weight, self.bias)


def get_inputs(
    batch_size: int = 16, num_features: int = 64, dim1: int = 256, dim2: int = 256
):
    x = torch.randn(batch_size, num_features, dim1, dim2)
    return [x]


input_names = ["x"]

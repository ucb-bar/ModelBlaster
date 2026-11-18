import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    """
    Functional Group Normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).
        weight (torch.Tensor): Weight tensor of shape (num_features).
        bias (torch.Tensor): Bias tensor of shape (num_features).
        num_groups (int): Number of groups to divide the channels into.
        eps (float): Epsilon parameter for numerical stability.

    Returns:
        torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
    """
    return F.group_norm(x, num_groups=num_groups, weight=weight, bias=bias, eps=eps)


class Model(nn.Module):
    """
    Simple model that performs Group Normalization.
    """

    def __init__(self, num_features: int = 64, num_groups: int = 8, eps: float = 1e-5):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return fn(x, self.weight, self.bias, self.num_groups, self.eps)


def get_inputs(
    batch_size: int = 16, num_features: int = 64, dim1: int = 256, dim2: int = 256
):
    x = torch.randn(batch_size, num_features, dim1, dim2)
    return [x]


input_names = ["x"]

import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    num_groups: int,
    hardtanh_min: float,
    hardtanh_max: float,
) -> torch.Tensor:
    """
    Applies linear layer, group normalization and hardtanh activation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features)
        bias (torch.Tensor): Bias vector of shape (out_features)
        group_norm_weight (torch.Tensor): Group norm weight of shape (out_features)
        group_norm_bias (torch.Tensor): Group norm bias of shape (out_features)
        num_groups (int): Number of groups for group normalization
        hardtanh_min (float): Minimum value for hardtanh
        hardtanh_max (float): Maximum value for hardtanh

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features)
    """
    x = F.linear(x, weight, bias)
    x = F.group_norm(x, num_groups, group_norm_weight, group_norm_bias)
    x = F.hardtanh(x, hardtanh_min, hardtanh_max)
    return x


class Model(nn.Module):
    """
    Simple model that performs a GEMM, applies Group Normalization, and then HardTanh.
    """

    def __init__(
        self,
        in_features: int = 1024,
        out_features: int = 512,
        num_groups: int = 8,
        hardtanh_min: float = -2.0,
        hardtanh_max: float = 2.0,
    ):
        super(Model, self).__init__()
        gemm = nn.Linear(in_features, out_features)
        group_norm = nn.GroupNorm(num_groups, out_features)
        self.weight = nn.Parameter(gemm.weight)
        self.bias = nn.Parameter(gemm.bias + torch.ones_like(gemm.bias) * 0.02)
        self.group_norm_weight = nn.Parameter(group_norm.weight)
        self.group_norm_bias = nn.Parameter(
            group_norm.bias + torch.ones_like(group_norm.bias) * 0.02
        )
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.weight,
            self.bias,
            self.group_norm_weight,
            self.group_norm_bias,
            self.num_groups,
            self.hardtanh_min,
            self.hardtanh_max,
        )


def get_inputs(batch_size: int = 128, in_features: int = 1024):
    x = torch.randn(batch_size, in_features)
    return [x]



input_names = ['x']

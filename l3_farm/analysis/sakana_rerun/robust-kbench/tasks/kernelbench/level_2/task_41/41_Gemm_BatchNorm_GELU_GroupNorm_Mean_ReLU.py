import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Model that performs a GEMM, BatchNorm, GELU, GroupNorm, Mean, and ReLU operations in sequence.
    """

    def __init__(self, in_features, out_features, num_groups):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.gemm.bias = nn.Parameter(
            self.gemm.bias + torch.randn_like(self.gemm.bias) * 0.02
        )

        self.batch_norm.weight = nn.Parameter(
            self.batch_norm.weight + torch.randn_like(self.batch_norm.weight) * 0.02
        )
        self.batch_norm.bias = nn.Parameter(
            self.batch_norm.bias + torch.randn_like(self.batch_norm.bias) * 0.02
        )
        self.batch_norm.running_mean = torch.randn(out_features)
        self.batch_norm.running_var = torch.randn(out_features).abs()

        self.group_norm.weight = nn.Parameter(
            self.group_norm.weight + torch.randn_like(self.group_norm.weight) * 0.02
        )
        self.group_norm.bias = nn.Parameter(
            self.group_norm.bias + torch.randn_like(self.group_norm.bias) * 0.02
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.batch_norm(x)
        x = torch.nn.functional.gelu(x)
        x = self.group_norm(x)
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.relu(x)
        return x


batch_size = 128
in_features = 512
out_features = 1024
num_groups = 8


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups]

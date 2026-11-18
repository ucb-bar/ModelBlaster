import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    bn_eps: float,
    bn_momentum: float,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    gemm_weight: torch.Tensor,
    gemm_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Performs matrix multiplication, batch normalization, scaling and softmax.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        running_mean (torch.Tensor): BatchNorm running mean
        running_var (torch.Tensor): BatchNorm running variance
        bn_eps (float): BatchNorm epsilon
        bn_momentum (float): BatchNorm momentum
        weight (torch.Tensor): BatchNorm weight parameter
        bias (torch.Tensor): BatchNorm bias parameter
        scale (torch.Tensor): Scale parameter
        gemm_weight (torch.Tensor): Linear layer weights
        gemm_bias (torch.Tensor): Linear layer bias

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features)
    """
    x = F.linear(x, gemm_weight, gemm_bias)

    if x.dim() == 2:
        x = F.batch_norm(
            x,
            running_mean,
            running_var,
            weight,
            bias,
            training=True,
            momentum=bn_momentum,
            eps=bn_eps,
        )
    else:
        raise ValueError("Expected 2D input tensor")

    x = scale * x
    x = F.softmax(x, dim=1)
    return x


class Model(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), Batch Normalization, scaling, and Softmax.
    """

    def __init__(
        self,
        in_features: int = 1024,
        out_features: int = 512,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ):
        super(Model, self).__init__()

        gemm = nn.Linear(in_features, out_features)
        self.gemm_weight = nn.Parameter(gemm.weight)
        self.gemm_bias = nn.Parameter(gemm.bias)

        batch_norm = nn.BatchNorm1d(out_features)
        self.bn_weight = nn.Parameter(
            batch_norm.weight + torch.randn(batch_norm.weight.shape) * 0.02
        )
        self.bn_bias = nn.Parameter(
            batch_norm.bias + torch.randn(batch_norm.bias.shape) * 0.02
        )
        self.register_buffer(
            "running_mean",
            batch_norm.running_mean + torch.randn(batch_norm.running_mean.shape) * 0.02,
        )
        self.register_buffer(
            "running_var",
            batch_norm.running_var
            + torch.randn(batch_norm.running_var.shape).abs() * 0.02,
        )

        scale_shape = (1,)
        self.scale = nn.Parameter(torch.randn(scale_shape) * 0.02)
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.running_mean,
            self.running_var,
            self.bn_eps,
            self.bn_momentum,
            self.bn_weight,
            self.bn_bias,
            self.scale,
            self.gemm_weight,
            self.gemm_bias,
        )


def get_inputs(batch_size: int = 128, in_features: int = 1024):
    x = torch.randn(batch_size, in_features)
    return [x]



input_names = ['x']

import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    eps: float,
    momentum: float,
    scale_factor: float,
    conv_transpose: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 3D transposed convolution, scaling, batch normalization and global average pooling.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
        eps (float): Small constant for numerical stability in batch norm
        momentum (float): Momentum for batch norm running stats
        conv_transpose (torch.Tensor): Transposed conv weights
        conv_transpose_bias (torch.Tensor): Transposed conv bias
        bn_weight (torch.Tensor): Batch norm weight parameter
        bn_bias (torch.Tensor): Batch norm bias parameter
        bn_running_mean (torch.Tensor): Batch norm running mean
        bn_running_var (torch.Tensor): Batch norm running variance

    Returns:
        torch.Tensor: Output tensor after applying operations
    """
    x = F.conv_transpose3d(x, conv_transpose, bias=conv_transpose_bias)
    x = x * scale_factor
    x = F.batch_norm(
        x,
        bn_running_mean,
        bn_running_var,
        bn_weight,
        bn_bias,
        training=True,
        momentum=momentum,
        eps=eps,
    )
    x = F.adaptive_avg_pool3d(x, (1, 1, 1))
    return x


class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, scales the output, applies batch normalization,
    and then performs global average pooling.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 32,
        kernel_size: int = 3,
        scale_factor: float = 2.0,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super(Model, self).__init__()
        conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.conv_transpose_parameter = nn.Parameter(conv.weight)
        self.conv_transpose_bias = nn.Parameter(conv.bias)

        bn = nn.BatchNorm3d(out_channels)
        self.bn_weight = nn.Parameter(bn.weight + torch.randn(bn.weight.shape) * 0.02)
        self.bn_bias = nn.Parameter(bn.bias + torch.randn(bn.bias.shape) * 0.02)
        self.register_buffer(
            "bn_running_mean",
            bn.running_mean + torch.randn(bn.running_mean.shape) * 0.02,
        )
        self.register_buffer(
            "bn_running_var",
            bn.running_var + torch.randn(bn.running_var.shape).abs() * 0.02,
        )
        self.scale_factor = scale_factor
        self.eps = eps
        self.momentum = momentum

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.eps,
            self.momentum,
            self.scale_factor,
            self.conv_transpose_parameter,
            self.conv_transpose_bias,
            self.bn_weight,
            self.bn_bias,
            self.bn_running_mean,
            self.bn_running_var,
        )


def get_inputs(
    batch_size: int = 16,
    in_channels: int = 64,
    depth: int = 16,
    height: int = 32,
    width: int = 32,
):
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]



input_names = ['x']

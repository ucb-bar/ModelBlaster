import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    out_channels: int,
    stride: int,
    padding: int,
    eps: float,
    scaling_factor: float,
    conv_transpose_weight: torch.Tensor,
    conv_transpose_bias: torch.Tensor,
    layer_norm_weight: torch.Tensor,
    layer_norm_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 3D transposed convolution, layer normalization, GELU activation and scaling.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        stride (int): Stride of the transposed convolution
        padding (int): Padding of the transposed convolution
        bias (bool): Whether to use bias in transposed convolution
        eps (float): Epsilon value for layer normalization
        scaling_factor (float): Factor to scale the output by
        conv_transpose_weight (torch.Tensor): Weight tensor for transposed convolution
        conv_transpose_bias (torch.Tensor): Bias tensor for transposed convolution
        layer_norm_weight (torch.Tensor): Weight tensor for layer normalization
        layer_norm_bias (torch.Tensor): Bias tensor for layer normalization

    Returns:
        torch.Tensor: Output tensor after applying operations
    """
    x = F.conv_transpose3d(
        x,
        conv_transpose_weight,
        bias=conv_transpose_bias,
        stride=stride,
        padding=padding,
    )

    x = F.layer_norm(
        x, (out_channels,), weight=layer_norm_weight, bias=layer_norm_bias, eps=eps
    )

    x = F.gelu(x)
    x = x * scaling_factor
    return x


class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, layer normalization, GELU activation, and scaling.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 64,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        eps: float = 1e-5,
        scaling_factor: float = 1.0,
    ):
        super(Model, self).__init__()
        conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.conv_transpose_weight = conv_transpose.weight
        self.conv_transpose_bias = nn.Parameter(
            conv_transpose.bias
            + torch.randn(
                conv_transpose.bias.shape,
                device=conv_transpose.bias.device,
                dtype=conv_transpose.bias.dtype,
            )
            * 0.02
        )
        self.layer_norm_weight = nn.Parameter(
            layer_norm.weight
            + torch.randn(
                layer_norm.weight.shape,
                device=layer_norm.weight.device,
                dtype=layer_norm.weight.dtype,
            )
            * 0.02
        )
        self.layer_norm_bias = nn.Parameter(
            layer_norm.bias
            + torch.randn(
                layer_norm.bias.shape,
                device=layer_norm.bias.device,
                dtype=layer_norm.bias.dtype,
            )
            * 0.02
        )
        self.stride = stride
        self.padding = padding
        self.eps = eps
        self.scaling_factor = scaling_factor
        self.out_channels = out_channels

    def forward(self, x, fn=forward_fn):
        return fn(
            x,
            self.out_channels,
            self.stride,
            self.padding,
            self.eps,
            self.scaling_factor,
            self.conv_transpose_weight,
            self.conv_transpose_bias,
            self.layer_norm_weight,
            self.layer_norm_bias,
        )


def get_inputs(
    batch_size: int = 128, in_channels: int = 32, D: int = 16, H: int = 32, W: int = 32
):
    x = torch.randn(batch_size, in_channels, D, H, W)
    return [x]



input_names = ['x']

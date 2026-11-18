import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    bn1_weight: torch.Tensor,
    bn1_bias: torch.Tensor,
    bn1_mean: torch.Tensor,
    bn1_var: torch.Tensor,
    conv2_weight: torch.Tensor,
    conv2_bias: torch.Tensor,
    bn2_weight: torch.Tensor,
    bn2_bias: torch.Tensor,
    bn2_mean: torch.Tensor,
    bn2_var: torch.Tensor,
    downsample_weight: torch.Tensor = None,
    downsample_bias: torch.Tensor = None,
    downsample_bn_weight: torch.Tensor = None,
    downsample_bn_bias: torch.Tensor = None,
    downsample_bn_mean: torch.Tensor = None,
    downsample_bn_var: torch.Tensor = None,
    stride: int = 1,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Implements a ResNet basic block with the following computation:

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        conv1_weight (torch.Tensor): Weights for first convolution
        conv1_bias (torch.Tensor): Bias for first convolution (or None)
        bn1_weight (torch.Tensor): BatchNorm1 weight
        bn1_bias (torch.Tensor): BatchNorm1 bias
        bn1_mean (torch.Tensor): BatchNorm1 running mean
        bn1_var (torch.Tensor): BatchNorm1 running variance
        conv2_weight (torch.Tensor): Weights for second convolution
        conv2_bias (torch.Tensor): Bias for second convolution (or None)
        bn2_weight (torch.Tensor): BatchNorm2 weight
        bn2_bias (torch.Tensor): BatchNorm2 bias
        bn2_mean (torch.Tensor): BatchNorm2 running mean
        bn2_var (torch.Tensor): BatchNorm2 running variance
        downsample_weight (torch.Tensor, optional): Weights for downsample convolution
        downsample_bias (torch.Tensor, optional): Bias for downsample convolution
        downsample_bn_weight (torch.Tensor, optional): BatchNorm weight for downsample
        downsample_bn_bias (torch.Tensor, optional): BatchNorm bias for downsample
        downsample_bn_mean (torch.Tensor, optional): BatchNorm running mean for downsample
        downsample_bn_var (torch.Tensor, optional): BatchNorm running variance for downsample
        stride (int): Stride for the first convolution. Default: 1
        eps (float): BatchNorm epsilon. Default: 1e-5

    Returns:
        torch.Tensor: Output tensor after applying the ResNet basic block
    """
    identity = x

    # First convolution block
    out = F.conv2d(x, conv1_weight, conv1_bias, stride=stride, padding=1)
    out = F.batch_norm(out, bn1_mean, bn1_var, bn1_weight, bn1_bias, False, 0.1, eps)
    out = F.relu(out)

    # Second convolution block
    out = F.conv2d(out, conv2_weight, conv2_bias, stride=1, padding=1)
    out = F.batch_norm(out, bn2_mean, bn2_var, bn2_weight, bn2_bias, False, 0.1, eps)

    # Downsample if needed
    if downsample_weight is not None:
        identity = F.conv2d(
            x, downsample_weight, downsample_bias, stride=stride, padding=0
        )
        identity = F.batch_norm(
            identity,
            downsample_bn_mean,
            downsample_bn_var,
            downsample_bn_weight,
            downsample_bn_bias,
            False,
            0.1,
            eps,
        )

    # Add residual connection and apply ReLU
    out += identity
    out = F.relu(out)

    return out


class Model(nn.Module):
    """
    Model that implements a ResNet basic block.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        stride: int = 1,
        downsample: bool = False,
    ):
        """
        Initializes the ResNet basic block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int): Stride for the first convolutional layer
            downsample (bool): Whether to use a downsample layer for the residual connection
        """
        super(Model, self).__init__()
        self.stride = stride
        self.eps = 1e-5

        # First convolution block
        conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.conv1_weight = nn.Parameter(conv1.weight.data.clone())
        self.conv1_bias = None

        bn1 = nn.BatchNorm2d(out_channels)
        self.bn1_weight = nn.Parameter(bn1.weight.data.clone())
        self.bn1_bias = nn.Parameter(bn1.bias.data.clone())
        self.register_buffer("bn1_mean", torch.zeros_like(bn1.running_mean))
        self.register_buffer("bn1_var", torch.ones_like(bn1.running_var))

        # Second convolution block
        conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv2_weight = nn.Parameter(conv2.weight.data.clone())
        self.conv2_bias = None

        bn2 = nn.BatchNorm2d(out_channels)
        self.bn2_weight = nn.Parameter(bn2.weight.data.clone())
        self.bn2_bias = nn.Parameter(bn2.bias.data.clone())
        self.register_buffer("bn2_mean", torch.zeros_like(bn2.running_mean))
        self.register_buffer("bn2_var", torch.ones_like(bn2.running_var))

        # Downsample layer
        self.has_downsample = (
            downsample or (stride != 1) or (in_channels != out_channels)
        )
        if self.has_downsample:
            downsample_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
            self.downsample_weight = nn.Parameter(downsample_conv.weight.data.clone())
            self.downsample_bias = None

            downsample_bn = nn.BatchNorm2d(out_channels)
            self.downsample_bn_weight = nn.Parameter(downsample_bn.weight.data.clone())
            self.downsample_bn_bias = nn.Parameter(downsample_bn.bias.data.clone())
            self.register_buffer(
                "downsample_bn_mean", torch.zeros_like(downsample_bn.running_mean)
            )
            self.register_buffer(
                "downsample_bn_var", torch.ones_like(downsample_bn.running_var)
            )
        else:
            self.downsample_weight = None
            self.downsample_bias = None
            self.downsample_bn_weight = None
            self.downsample_bn_bias = None
            self.downsample_bn_mean = None
            self.downsample_bn_var = None

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Forward pass that calls forward_fn.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            fn: Function to call, defaults to forward_fn

        Returns:
            torch.Tensor: Output tensor after applying the ResNet basic block
        """
        return fn(
            x,
            self.conv1_weight,
            self.conv1_bias,
            self.bn1_weight,
            self.bn1_bias,
            self.bn1_mean,
            self.bn1_var,
            self.conv2_weight,
            self.conv2_bias,
            self.bn2_weight,
            self.bn2_bias,
            self.bn2_mean,
            self.bn2_var,
            self.downsample_weight,
            self.downsample_bias,
            self.downsample_bn_weight,
            self.downsample_bn_bias,
            self.downsample_bn_mean,
            self.downsample_bn_var,
            self.stride,
            self.eps,
        )


def get_inputs(batch_size: int = 10, in_channels: int = 3):
    return [torch.randn(batch_size, in_channels, 224, 224)]


input_names = ["x"]

import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    weights: torch.Tensor,
    biases: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """Implements a 2D convolution layer with the following computation:
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        weights (torch.Tensor): Weights tensor of shape (out_channels, in_channels, kernel_size, kernel_size)
        biases (torch.Tensor): Biases vector of shape (out_channels)
        stride (int): Stride of the convolution. Default: 1
        padding (int): Padding added to all sides of the input. Default: 0

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out)
    """
    return F.conv2d(x, weights, biases, stride=stride, padding=padding)


class Model(nn.Module):
    """
    Simple model that performs Feedforward network block.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        init_method: str = "xavier",
    ):
        """
        Initializes the Conv2d block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the convolving kernel
            stride (int): Stride of the convolution
            padding (int, optional): Padding added to all sides of the input. 
                                   If None, padding will be kernel_size//2
            init_method (str): Weight initialization method
        """
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size//2 if padding is None else padding
        
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, 
                        stride=self.stride, padding=self.padding)
        import math

        if init_method == "kaiming":
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
        elif init_method == "xavier":
            nn.init.xavier_normal_(conv.weight)
        elif init_method == "normal":
            nn.init.normal_(conv.weight)
        nn.init.zeros_(conv.bias)

        self.weights = nn.Parameter(conv.weight.data.clone())
        self.biases = nn.Parameter(conv.bias.data.clone())

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Forward pass that calls forward_fn.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            fn: Function to call, defaults to forward_fn

        Returns:
            torch.Tensor: Output tensor with shape determined by stride and padding settings
        """
        return fn(x, self.weights, self.biases, self.stride, self.padding)


def get_inputs(
    batch_size: int = 16,
    in_channels: int = 64,
    height: int = 32,
    width: int = 32,
):
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

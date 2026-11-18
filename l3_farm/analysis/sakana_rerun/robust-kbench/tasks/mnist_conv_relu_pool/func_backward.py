import torch
import torch.nn as nn
import torch.nn.functional as F


class AutogradFunction(torch.autograd.Function):
    backward_fn = None

    @staticmethod
    def forward(ctx, x, weights, biases):
        # Save inputs for backward pass
        ctx.save_for_backward(x, weights, biases)
        # Apply 2D convolution
        x = F.conv2d(x, weights, bias=biases)
        # Apply ReLU activation
        x = F.relu(x)
        # Apply max pooling
        x = F.max_pool2d(x, kernel_size=2)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved inputs
        x, weights, biases = ctx.saved_tensors

        # Use the class-level backward function
        grad_input, grad_weights, grad_biases = AutogradFunction.backward_fn(
            x,
            weights,
            biases,
            grad_output,
        )
        return grad_input, grad_weights, grad_biases


def forward_fn(
    x: torch.Tensor,
    weights: torch.Tensor,
    biases: torch.Tensor,
) -> torch.Tensor:
    """Implements a 2D convolutional layer with ReLU activation and max-pooling with kernel size 2:

    y = conv2d(x, W) + b
    y = relu(y)
    y = max_pool2d(y)

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        weights (torch.Tensor): Weights matrix of shape (out_channels, in_channels, kernel_height, kernel_width)
        biases (torch.Tensor): Biases vector of shape (out_channels)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width)
    """
    # Apply 2D convolution
    x = F.conv2d(x, weights, bias=biases)
    # Apply ReLU activation
    x = F.relu(x)
    # Apply max pooling
    x = F.max_pool2d(x, kernel_size=2)
    return x


class Model(nn.Module):
    """
    Simple model that performs a Conv2D layer with ReLU activation and max-pooling with kernel size 2.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 32,
        kernel_size: int = 3,
        stride: int = 1,
        init_method: str = "normal",
    ):
        """
        Initializes the Conv2D layer with ReLU activation and max-pooling with kernel size 2.

        Args:
            num_features (int): Number of features in the input and output tensors
            up_features (int): Number of features in the up projection
        """
        super(Model, self).__init__()
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
        )
        import math

        if init_method == "kaiming":
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
        elif init_method == "xavier":
            nn.init.xavier_normal_(conv.weight)
        elif init_method == "normal":
            nn.init.normal_(conv.weight)
        # Initialize biase with random non-zero values
        nn.init.normal_(conv.bias, mean=0.0, std=0.1)

        self.weights = nn.Parameter(conv.weight.data.clone())
        self.biases = nn.Parameter(conv.bias.data.clone())

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Forward pass that calls forward_fn.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_input_features)
            fn: Function to call, defaults to forward_fn

        Returns:
            torch.Tensor: Output of module_fn of shape (batch_size, num_tokens, num_features)
        """
        return fn(x, self.weights, self.biases)


def get_inputs(
    batch_size: int = 64,
    in_channels: int = 1,
    height: int = 28,
    width: int = 28,
):
    x = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
    return [x]


input_names = ["x"]

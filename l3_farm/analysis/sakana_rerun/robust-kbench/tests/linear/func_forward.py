import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    weights: torch.Tensor,
    biases: torch.Tensor,
) -> torch.Tensor:
    """Implements a linear layer with the following computation:

    y = x @ W^T + b

    where @ denotes matrix multiplication, W^T is the transpose of the weights matrix,
    and b is the bias vector that gets broadcast across the batch dimension.
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features)
        weights (torch.Tensor): Weights matrix of shape (num_features, num_features)
        biases (torch.Tensor): Biases vector of shape (num_features)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_features)
    """
    return F.linear(x, weights, biases)


class Model(nn.Module):
    """
    Simple model that performs Feedforward network block.
    """

    def __init__(
        self,
        num_input_features: int = 4096,
        num_output_features: int = 4096,
    ):
        """
        Initializes the Feedforward network block.

        Args:
            num_features (int): Number of features in the input and output tensors
            up_features (int): Number of features in the up projection
        """
        super(Model, self).__init__()
        linear = nn.Linear(num_input_features, num_output_features)
        self.weights = nn.Parameter(linear.weight.data.clone())
        self.biases = nn.Parameter(linear.bias.data.clone())

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
    batch_size: int = 16,
    num_input_features: int = 4096,
):
    x = torch.randn(batch_size, num_input_features)
    return [x]

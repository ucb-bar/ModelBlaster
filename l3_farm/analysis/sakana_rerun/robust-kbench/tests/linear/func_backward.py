import torch
import torch.nn as nn
import torch.nn.functional as F


class AutogradFunction(torch.autograd.Function):
    backward_fn = None

    @staticmethod
    def forward(ctx, x, weights, biases):
        # Save inputs for backward pass
        ctx.save_for_backward(x, weights)
        return F.linear(x, weights, biases)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved inputs
        x, weights = ctx.saved_tensors

        # Use the class-level backward function
        grad_input, grad_weights, grad_biases = AutogradFunction.backward_fn(
            grad_output, x, weights
        )
        return grad_input, grad_weights, grad_biases


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


class Model(torch.nn.Module):
    def __init__(
        self,
        num_input_features: int = 4096,
        num_output_features: int = 4096,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(num_input_features, num_output_features)
        # Initialize parameters with requires_grad=True
        self.weights = nn.Parameter(
            self.linear.weight.data.clone(),
            requires_grad=True,
        )
        self.biases = nn.Parameter(
            self.linear.bias.data.clone(),
            requires_grad=True,
        )

    def forward(self, x, fn=forward_fn):
        return fn(x, self.weights, self.biases)


def get_inputs(
    batch_size: int = 16,
    num_input_features: int = 4096,
):
    x = torch.randn(batch_size, num_input_features)
    return [x]

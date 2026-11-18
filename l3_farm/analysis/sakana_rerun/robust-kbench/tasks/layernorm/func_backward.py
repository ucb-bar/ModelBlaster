import torch
import torch.nn as nn
import torch.nn.functional as F


class AutogradFunction(torch.autograd.Function):
    backward_fn = None

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        # Save inputs for backward pass
        normalized_shape = tuple(x.shape[-len(weight.shape) :])
        output = F.layer_norm(x, normalized_shape, weight, bias, eps)
        ctx.save_for_backward(x, weight, bias)
        ctx.eps = eps
        ctx.normalized_shape = normalized_shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved inputs
        x, weight, bias = ctx.saved_tensors
        eps = ctx.eps
        normalized_shape = ctx.normalized_shape

        # Use the class-level backward function
        grad_input, grad_weight, grad_bias = AutogradFunction.backward_fn(
            grad_output, x, weight, bias, eps, normalized_shape
        )
        return grad_input, grad_weight, grad_bias, None


def forward_fn(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    """
    Functional implementation of LayerNorm. Layer normalization computes the mean and variance across the last N dimensions specified by normalized_shape. For an input x, the formula is:

        y = (x - E[x]) / sqrt(Var[x] + eps) * weight + bias

    where E[x] and Var[x] are computed across the normalized dimensions. The weight and bias parameters are learnable affine transformations applied after normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (*, normalized_shape).
        weight (torch.Tensor): Weight tensor of shape (normalized_shape).
        bias (torch.Tensor): Bias tensor of shape (normalized_shape).
        eps (float): Epsilon parameter for numerical stability.

    Returns:
        torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
    """
    # Get the normalized shape from the weight tensor
    normalized_shape = tuple(x.shape[-len(weight.shape) :])
    return F.layer_norm(
        x, normalized_shape=normalized_shape, weight=weight, bias=bias, eps=eps
    )


class Model(nn.Module):
    """
    Simple model that performs Layer Normalization.
    """

    def __init__(
        self,
        num_features: int,
        dim1: int,
        dim2: int,
        eps: float = 1e-5,
        init_method: str = "standard",
    ):
        """
        Initializes the LayerNorm layer parameters.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(Model, self).__init__()
        self.normalized_shape = (num_features, dim1, dim2)
        self.eps = eps

        if init_method == "standard":
            weight = torch.ones(self.normalized_shape)
            bias = torch.zeros(self.normalized_shape)
        elif init_method == "random":
            weight = torch.randn(self.normalized_shape)
            bias = torch.randn(self.normalized_shape)

        self.weight = nn.Parameter(
            weight,
            requires_grad=True,
        )
        self.bias = nn.Parameter(
            bias,
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).
            fn: Function to apply (defaults to forward_fn)

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return fn(x, self.weight, self.bias, self.eps)


def get_inputs(
    batch_size: int = 16, num_features: int = 64, dim1: int = 256, dim2: int = 256
):
    x = torch.randn(batch_size, num_features, dim1, dim2, requires_grad=True)
    return [x]


input_names = ["x"]

import torch
import torch.nn as nn


class AutogradFunction(torch.autograd.Function):
    backward_fn = None

    @staticmethod
    def forward(ctx, x, w, eps=1e-8):
        # Save inputs for backward pass
        ctx.save_for_backward(x, w)
        ctx.eps = eps

        # Convert to fp32 for better numerical stability
        x_fp32 = x.float()
        # Calculate RMS
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
        # Normalize and convert back to input dtype
        x_normed = (x_fp32 * rms).type_as(x)
        # Apply weight
        output = x_normed * w
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        eps = ctx.eps

        # Use the class-level backward function
        grad_input, grad_weight = AutogradFunction.backward_fn(grad_output, x, w, eps)
        return grad_input, grad_weight, None


def forward_fn(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Applies RMS (Root Mean Square) Normalization to the input tensor. RMSNorm normalizes by the root mean square:

    RMSNorm(x) = x / RMS(x)

    where RMS(x) = sqrt(mean(x^2) + eps)

    Note that torch.rsqrt is numerically stable, while 1.0/torch.sqrt is not.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, *)
        w (torch.Tensor): Weight tensor of shape (num_features,)
        eps (float): Small value added to denominator for numerical stability

    Returns:
        torch.Tensor: Output tensor with RMS Normalization applied
    """
    x_fp32 = x.float()
    x_normed = (
        x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
    ).type_as(x)
    return x_normed * w


class Model(nn.Module):
    """
    Simple model that performs RMS Normalization.
    """

    def __init__(self, num_features: int, eps: float):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor
            eps (float): Small value added to denominator for numerical stability
        """
        super(Model, self).__init__()
        rms = nn.RMSNorm(normalized_shape=[num_features])
        rms.weight.data.normal_(mean=1.0, std=0.1)
        self.w = nn.Parameter(rms.weight.data.clone(), requires_grad=True)
        # sample from normal distribution
        self.eps = eps

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Forward pass that calls forward_fn.

        Args:
            x (torch.Tensor): Input tensor
            fn: Function to call, defaults to forward_fn

        Returns:
            torch.Tensor: Output of forward_fn
        """
        return fn(x, self.w, self.eps)


def get_inputs(
    batch_size: int = 16,
    num_tokens: int = 1024,
    num_features: int = 4096,
):
    x = torch.randn(batch_size, num_tokens, num_features, requires_grad=True)
    return [x]


input_names = ["x"]

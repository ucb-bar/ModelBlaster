import torch
import torch.nn as nn


def forward_fn(x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Applies RMS Normalization to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, *)
        eps (float): Small value added to denominator for numerical stability

    Returns:
        torch.Tensor: Output tensor with RMS Normalization applied
    """
    rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)
    return x / rms


class Model(nn.Module):
    """
    Simple model that performs RMS Normalization.
    """

    def __init__(self, num_features: int = 64, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor
            eps (float): Small value added to denominator for numerical stability
        """
        super(Model, self).__init__()
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
        return fn(x, self.eps)


def get_inputs(
    batch_size: int = 16, num_features: int = 64, dim1: int = 256, dim2: int = 256
):
    x = torch.randn(batch_size, num_features, dim1, dim2)
    return [x]



input_names = ['x']

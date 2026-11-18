import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the Mean Squared Error loss for regression tasks.

    Args:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values.

    Returns:
        torch.Tensor: Mean Squared Error loss.
    """
    return F.mse_loss(predictions, targets, reduction="mean")


class Model(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks.

    Parameters:
        None
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets, fn=forward_fn):
        return fn(predictions, targets)


def get_inputs(batch_size: int = 128, input_shape: int = 4096):
    predictions = torch.randn(batch_size, *(input_shape,))
    targets = torch.randn(batch_size, *(input_shape,))
    return [predictions, targets]


input_names = ["predictions", "targets"]

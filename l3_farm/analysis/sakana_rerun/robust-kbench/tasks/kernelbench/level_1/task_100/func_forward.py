import torch
import torch.nn as nn


def forward_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the Hinge Loss for binary classification tasks.

    Args:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values.

    Returns:
        torch.Tensor: Hinge Loss.
    """
    return torch.mean(torch.clamp(1 - predictions * targets, min=0))


class Model(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.

    Parameters:
        None
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets, fn=forward_fn):
        return fn(predictions, targets)


def get_inputs(batch_size: int = 128, input_shape: int = 1):
    predictions = torch.randn(batch_size, *(input_shape,))
    targets = torch.randint(0, 2, (batch_size,)).float() * 2 - 1
    return [predictions, targets]


input_names = ["predictions", "targets"]

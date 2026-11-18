import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the Cross Entropy Loss for multi-class classification tasks.

    Args:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values.

    Returns:
        torch.Tensor: Cross Entropy Loss.
    """
    return F.cross_entropy(predictions, targets)


class Model(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks.

    Parameters:
        None
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets, fn=forward_fn):
        return fn(predictions, targets)


def get_inputs(batch_size: int = 4096, num_classes: int = 10):
    predictions = torch.randn(batch_size, *(num_classes,))
    targets = torch.randint(0, num_classes, (batch_size,))
    return [predictions, targets]


input_names = ["predictions", "targets"]

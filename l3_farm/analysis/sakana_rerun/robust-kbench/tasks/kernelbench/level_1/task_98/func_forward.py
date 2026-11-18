import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kullback-Leibler Divergence for comparing two distributions.

    Args:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values.

    Returns:
        torch.Tensor: Kullback-Leibler Divergence.
    """
    return F.kl_div(torch.log(predictions), targets, reduction="batchmean")


class Model(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.

    Parameters:
        None
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets, fn=forward_fn):
        return fn(predictions, targets)


def get_inputs(batch_size: int = 128, input_shape: int = 4096):
    predictions = torch.randn(batch_size, *(input_shape,)).softmax(dim=-1)
    targets = torch.randn(batch_size, *(input_shape,)).softmax(dim=-1)
    return [predictions, targets]


input_names = ["predictions", "targets"]

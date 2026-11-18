import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float
) -> torch.Tensor:
    """
    Computes the Triplet Margin Loss for metric learning tasks.

    Args:
        anchor (torch.Tensor): Anchor values.
        positive (torch.Tensor): Positive values.
        negative (torch.Tensor): Negative values.
        margin (float): Margin value.

    Returns:
        torch.Tensor: Triplet Margin Loss.
    """
    return F.triplet_margin_loss(anchor, positive, negative, margin=margin)


class Model(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.
    """

    def __init__(self, margin: float = 1.0):
        super(Model, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, fn=forward_fn):
        return fn(anchor, positive, negative, self.margin)


def get_inputs(batch_size: int = 128, input_shape: int = 4096):
    anchor = torch.randn(batch_size, *(input_shape,))
    positive = torch.randn(batch_size, *(input_shape,))
    negative = torch.randn(batch_size, *(input_shape,))
    return [anchor, positive, negative]


input_names = ["anchor", "positive", "negative"]

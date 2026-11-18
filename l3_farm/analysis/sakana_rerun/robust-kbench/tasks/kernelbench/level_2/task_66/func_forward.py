import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    dropout_p: float,
    training: bool,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Performs matrix multiplication, applies dropout, calculates mean, and applies softmax.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        dropout_p (float): Dropout probability
        training (bool): Whether model is in training mode
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features)
        bias (torch.Tensor): Bias vector of shape (out_features)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features)
    """
    x = F.linear(x, weight, bias)
    x = F.dropout(x, p=dropout_p, training=training)
    x = torch.mean(x, dim=1, keepdim=True)
    x = F.softmax(x, dim=1)
    return x


class Model(nn.Module):
    """
    A model that performs matrix multiplication, applies dropout, calculates the mean, and then applies softmax.
    """

    def __init__(
        self,
        in_features: int = 100,
        out_features: int = 50,
        dropout_p: float = 0.2,
        training: bool = True,
    ):
        super(Model, self).__init__()
        mm = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(mm.weight)
        self.bias = nn.Parameter(mm.bias)
        self.dropout_p = dropout_p
        self.training = training

    def forward(self, x, fn=forward_fn):
        return fn(x, self.dropout_p, self.training, self.weight, self.bias)


def get_inputs(batch_size: int = 128, in_features: int = 100):
    x = torch.randn(batch_size, in_features)
    return [x]



input_names = ['x']

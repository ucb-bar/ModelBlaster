import torch
import torch.nn as nn
import torch.nn.functional as F


class AutogradFunction(torch.autograd.Function):
    backward_fn = None

    @staticmethod
    def forward(ctx, predictions, targets):
        # Save inputs for backward pass
        ctx.save_for_backward(predictions, targets)
        return F.cross_entropy(predictions, targets, reduction="mean")

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved inputs
        predictions, targets = ctx.saved_tensors

        # Use the class-level backward function
        grad_predictions = AutogradFunction.backward_fn(
            grad_output, predictions, targets
        )
        return grad_predictions, None


def forward_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the Cross Entropy Loss for multi-class classification tasks.

    The cross entropy loss combines log softmax and negative log likelihood loss. For input x and target class y, it computes:

    loss = -log(softmax(x)[y])

    where softmax(x)[i] = exp(x[i]) / sum_j(exp(x[j]))

    This measures the dissimilarity between the predicted probability distribution and the true distribution (one-hot encoded target).

    Args:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values.

    Returns:
        torch.Tensor: Cross Entropy Loss.
    """
    return F.cross_entropy(predictions, targets, reduction="mean")


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
    return [
        torch.randn(batch_size, *(num_classes,), requires_grad=True),
        torch.randint(0, num_classes, (batch_size,)),
    ]


input_names = ["predictions", "targets"]

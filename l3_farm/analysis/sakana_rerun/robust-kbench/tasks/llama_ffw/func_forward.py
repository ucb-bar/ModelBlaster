import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_fn(
    x: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> torch.Tensor:
    """Feedforward network block for a Transformer.
    Implements the feedforward block from LLaMA which consists of:
    1. Two parallel linear projections (gate_proj and up_proj) from num_features to up_features
    2. SiLU activation on the gate projection
    3. Element-wise multiplication of the activated gate with the up projection
    4. Linear projection back to num_features dimension

    The computation can be expressed mathematically as:

    gate = SiLU(x @ gate_proj)
    up = x @ up_proj
    down = (gate * up) @ down_proj

    where @ denotes matrix multiplication and * denotes element-wise multiplication.

    This is a variant of the standard MLP layer that uses gating to control information flow. The SiLU activation and gating mechanism help with training stability and expressiveness.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, num_features)
        gate_proj (torch.Tensor): Gate projection matrix of shape (num_features, up_features)
        up_proj (torch.Tensor): Up projection matrix of shape (num_features, up_features)
        down_proj (torch.Tensor): Down projection matrix of shape (up_features, num_features)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_tokens, num_features)
    """
    gate = F.linear(x, gate_proj)
    gate = F.silu(gate)
    up = F.linear(x, up_proj)
    up = gate * up
    down = F.linear(up, down_proj)
    return down


class Model(nn.Module):
    """
    Simple model that performs Feedforward network block.
    """

    def __init__(
        self,
        num_features: int = 4096,
        up_features: int = 14336,
    ):
        """
        Initializes the Feedforward network block.

        Args:
            num_features (int): Number of features in the input and output tensors
            up_features (int): Number of features in the up projection
        """
        super(Model, self).__init__()
        l_gate = nn.Linear(num_features, up_features)
        l_up = nn.Linear(num_features, up_features)
        l_down = nn.Linear(up_features, num_features)
        self.gate_proj = nn.Parameter(l_gate.weight.data.clone())
        self.up_proj = nn.Parameter(l_up.weight.data.clone())
        self.down_proj = nn.Parameter(l_down.weight.data.clone())

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Forward pass that calls forward_fn.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, num_features)
            fn: Function to call, defaults to forward_fn

        Returns:
            torch.Tensor: Output of module_fn of shape (batch_size, num_tokens, num_features)
        """
        return fn(x, self.gate_proj, self.up_proj, self.down_proj)


def get_inputs(
    batch_size: int = 16,
    num_tokens: int = 1024,
    num_features: int = 4096,
):
    x = torch.randn(batch_size, num_tokens, num_features)
    return [x]


input_names = ["x"]

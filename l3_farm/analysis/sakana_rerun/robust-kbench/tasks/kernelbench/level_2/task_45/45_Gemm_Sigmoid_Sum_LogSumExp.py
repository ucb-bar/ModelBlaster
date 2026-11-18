import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), applies Sigmoid, sums the result, and calculates the LogSumExp.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear1.bias = nn.Parameter(self.linear1.bias + torch.randn(self.linear1.bias.shape, device=self.linear1.bias.device, dtype=self.linear1.bias.dtype) * 0.02)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = torch.sum(x, dim=1)
        x = torch.logsumexp(x, dim=0)
        return x

batch_size = 128
input_size = 10
hidden_size = 20
output_size = 5

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, output_size]
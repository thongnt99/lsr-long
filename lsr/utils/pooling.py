import torch
from torch import nn


class MaxPoolValue(nn.Module):
    """Max pooling over a specified dimension"""

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.max(inputs, dim=self.dim).values


class SumPool(nn.Module):
    """Sum pooling over a specified dimension"""

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.sum(inputs, dim=self.dim)


class AbsPool(nn.Module):
    """Pool by maximum absolute value"""

    def __init__(self, dim=1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        abs_inputs = torch.abs(inputs)
        max_index = abs_inputs.argmax(dim=self.dim, keepdim=True)
        max_abs_value = inputs.gather(1, max_index).squeeze(1)
        return max_abs_value

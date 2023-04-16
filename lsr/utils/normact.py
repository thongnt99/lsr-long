"""Contains common normalization or activation function wrapped in class"""
from torch import nn
import torch


class Log1P(nn.Module):
    """This is a class warper of torch.log1p function"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs):
        return torch.log1p(inputs)


class NoNorm(nn.Module):
    """This module return the inputs itself"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs):
        return inputs


class AllOne(nn.Module):
    """This module return 1.0 for any inputs"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs):
        return 1.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class LinearRegressionModel(nn.Module):
    """A simple linear regression model using PyTorch.

    Args:
        nn (torch.nn.Module): Inherits from PyTorch's nn.Module.
    """

    def __init__(self, input_size: int, output_size: int, bias=True):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x):
        out = self.linear(x)
        return out

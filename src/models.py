import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        """
        Initializes the MLP model.

        Args:
            input_size (int): Size of the input layer.
            hidden_sizes (List[int]): List of sizes for the hidden layers.
            output_size (int): Size of the output layer.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        self.fc2 = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the MLP.
        """
        x = F.relu(self.fc1(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.fc2(x)
        return x

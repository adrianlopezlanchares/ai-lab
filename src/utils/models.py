import torch
from torch import nn

from typing import List


class MLP(nn.Module):
    """Generic Multi-Layer Perceptron (MLP) model for building parameter prediction.

    Attributes:
        model (nn.Sequential): A sequential container of layers that defines the MLP architecture.
        It consists of multiple linear layers followed by ReLU activation functions.
    """

    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        """Initializes the MLP model with specified input, hidden, and output sizes.

        Args:
            input_size (int): Number of input features.
            hidden_sizes (List[int]): List of integers representing the number of neurons in each hidden layer.
            output_size (int): Number of output features.
        """
        super(MLP, self).__init__()
        layers = []
        current_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size

        layers.append(nn.Linear(current_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the MLP.
        """
        return self.model(x)


class LinearRegressionModel(nn.Module):
    """A simple linear regression model using PyTorch.

    Attributes:
        linear (nn.Linear): A linear layer that performs the linear transformation.
    """

    def __init__(self, input_size: int, output_size: int, bias=True):
        """Initializes the LinearRegressionModel.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            bias (bool, optional): Whether to include a bias term in the linear layer. Defaults to True.
        """
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after applying the linear transformation.
        """
        out = self.linear(x)
        return out

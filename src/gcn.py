# kcn.py: a file containing all neural network definitions for Graph Convolutional Network (GCN)

import torch
from torch import nn
from torch_geometry.nn import GCNConv, Sequential
from typing import List, Union

class Erf(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.erf(x)

ACTIVATIONS = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "softmax": nn.Softmax(),
    "erf": Erf(),
}


class GCN(nn.Module):
    """
    Defines a Graph Convolutional Network (GCN)

    Arguments
    --------------
    input_dim: int, the dimension of input tensor
    num_hidden_layers: int, the number of hidden layers
    hidden_dims: int or list of int, the dimensions of hidden layers
    activation: str, the activation function to use, default to "relu"
    """
    def __init__(self, input_dim: int, num_hidden_layers: int, hidden_dims: Union[int, List[int]], 
                activation: str = "relu") -> None:
        super(GCN, self).__init__()
        self.input_dim: int = input_dim
        self.num_hidden_layers: int = num_hidden_layers
        self.hidden_dims: Union[int, List[int]] = hidden_dims
        self.activation: str = activation
        self._validate_inputs()
        self._build_layers()

    def _validate_inputs(self) -> None:
        if isinstance(self.hidden_dims, List):
            assert len(self.hidden_dims) == self.num_hidden_layers, \
                "Number of hidden layers must match the length of hidden dimensions"
        assert self.activation in ACTIVATIONS, "Activation function must be one of {}".format(ACTIVATIONS.keys())
    
    def _build_layers(self) -> None:
        """
        Build the hidden layers
        """
        layers = []
        if isinstance(self.hidden_dims, int):
            hidden_dims = [self.hidden_dims] * self.num_hidden_layers
        else:
            hidden_dims = self.hidden_dims
        for i in range(self.num_hidden_layers):
            if i == 0:
                layers.append(GCNConv(self.input_dim, hidden_dims[i]))
            else:
                layers.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
            layers.append(ACTIVATIONS[self.activation])
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.gcn_layers = Sequential(*layers)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.gcn_layers(x, edge_index)
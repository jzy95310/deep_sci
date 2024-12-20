# gcn.py: a file containing all neural network definitions for Graph Convolutional Network (GCN)

import torch
from torch import nn
from torch_geometric.nn import GraphConv, Sequential
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
                layers.append((GraphConv(self.input_dim, hidden_dims[i]), 'x, edge_indices -> x'))
            else:
                layers.append((GraphConv(hidden_dims[i-1], hidden_dims[i]), 'x, edge_indices -> x'))
            layers.append(ACTIVATIONS[self.activation])
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.gcn_layers = Sequential('x, edge_indices', layers)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.gcn_layers(x, edge_index)
    
    def initialize_weights(self, method: str = 'kaiming') -> None:
        """
        Initialize weights of the model

        Arguments
        --------------
        method: str, the method to use for weight initialization, default to "kaiming"
        """
        if method == 'kaiming':
            for layer in self.gcn_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
                elif isinstance(layer, GraphConv):
                    for param in layer.parameters():
                        if len(param.shape) > 1:
                            nn.init.kaiming_normal_(param)
        elif method == 'xavier':
            for layer in self.gcn_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
                elif isinstance(layer, GraphConv):
                    for param in layer.parameters():
                        if len(param.shape) > 1:
                            nn.init.xavier_normal_(param)
        else:
            raise ValueError("Invalid method for weight initialization")

# ########################################################################################
# MIT License

# Copyright (c) 2023 Ziyang Jiang

# Permission is hereby granted, free of charge, to any person obtaining a copy of this 
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, 
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
# permit persons to whom the Software is furnished to do so, subject to the following 
# conditions:

# The above copyright notice and this permission notice shall be included in all copies 
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE 
# OR OTHER DEALINGS IN THE SOFTWARE.
# ########################################################################################
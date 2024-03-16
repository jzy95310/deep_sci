# networks.py: a file containing all neural network definitions for nonlinear spatial causal inference model
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import torch
from torch import nn
import numpy as np
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


class DeepKrigingEmbedding2D(nn.Module):
    """
    Two-dimensional embedding layer for DeepKriging
    """
    def __init__(self, K: int):
        super(DeepKrigingEmbedding2D, self).__init__()
        self.K = K
        self.num_basis = [(9*2**(h-1)+1)**2 for h in range(1,self.K+1)]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: torch.Tensor, shape (N,2), the spatial information
        """
        knots_1d = [torch.linspace(0, 1, int(np.sqrt(i))).to(self.device) for i in self.num_basis]
        N = s.shape[0]
        phi = torch.zeros(N, sum(self.num_basis)).to(self.device)
        K = 0
        for res, num_basis_res in enumerate(self.num_basis):
            theta = 1 / np.sqrt(num_basis_res) * 2.5
            knots_s1, knots_s2 = torch.meshgrid(knots_1d[res], knots_1d[res], indexing='ij')
            knots = torch.stack((knots_s1.flatten(), knots_s2.flatten()), dim=1).to(self.device)
            d = torch.cdist(s, knots) / theta
            mask = (d >= 0) & (d <= 1)
            weights = torch.zeros_like(d)
            weights[mask] = ((1 - d[mask]) ** 6 * (35 * d[mask] ** 2 + 18 * d[mask] + 3) / 3)
            phi[:, K:K + num_basis_res] = weights
            K += num_basis_res
        return phi


class MLP(nn.Module):
    """
    Defines a multi-layer perceptron

    Arguments
    --------------
    input_dim: int, the dimension of input tensor
    num_hidden_layers: int, the number of hidden layers
    hidden_dims: int or list of int, the dimensions of hidden layers
    batch_norm: bool, whether to use batch normalization in each hidden layer, default to False
    p_dropout: float, the dropout probability, default to 0.0
    activation: str, the activation function to use, default to "relu"
    """
    def __init__(self, input_dim: int, num_hidden_layers: int, hidden_dims: Union[int, List[int]], 
                 batch_norm: bool = False, p_dropout: float = 0.0, activation: str = "relu") -> None:
        super(MLP, self).__init__()
        self.input_dim: int = input_dim
        self.num_hidden_layers: int = num_hidden_layers
        self.hidden_dims: Union[int, List[int]] = hidden_dims
        self.batch_norm: bool = batch_norm
        self.p_dropout: float = p_dropout
        self.activation: str = activation
        self._validate_inputs()
        self._build_layers()
    
    def _validate_inputs(self) -> None:
        if isinstance(self.hidden_dims, List):
            assert len(self.hidden_dims) == self.num_hidden_layers, \
                "Number of hidden layers must match the length of hidden dimensions"
        assert self.p_dropout >= 0 and self.p_dropout < 1, "Dropout probability must be in [0, 1)"
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
                layers.append(nn.Linear(self.input_dim, hidden_dims[i]))
            else:
                layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(ACTIVATIONS[self.activation])
            if self.p_dropout > 0:
                layers.append(nn.Dropout(self.p_dropout))
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.mlp_layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_layers(x)


class ConvNet(nn.Module):
    """
    Defines a convolutional neural network for processing spatial confounders

    Arguments
    --------------
    input_width: int, the width of input tensor
    input_height: int, the height of input tensor
    num_channels: int, number of input channels, default to 1
    kernel_size: int, the initial kernel size of each convolutional layer, default to 7
    stride: int, the initial stride of each convolutional layer, default to 3
    intermediate_channels: int, the maximum number of channels in the convolutional layers, default to 64
    dense_hidden_dim: int or list of int, the dimension of hidden layers after convolutional layers, default to 128
    batch_norm: bool, whether to use batch normalization in each convolutional block, default to False
    p_dropout: float, the dropout probability, default to 0.0
    activation: str, the activation function to use, default to "relu"
    """
    def __init__(self, input_width: int, input_height: int, num_channels: int = 1, kernel_size: int = 7, 
                 stride: int = 3, intermediate_channels: int = 64, dense_hidden_dim: int = 128, batch_norm: bool = False, 
                 p_dropout: float = 0.0, activation: str = "relu") -> None:
        super(ConvNet, self).__init__()
        self.input_width: int = input_width
        self.input_height: int = input_height
        self.num_channels: int = num_channels
        self.kernel_size: int = kernel_size
        self.stride: int = stride
        self.intermediate_channels: int = intermediate_channels
        self.dense_hidden_dim: int = dense_hidden_dim
        self.batch_norm: bool = batch_norm
        self.p_dropout: float = p_dropout
        self.activation: str = activation
        self._validate_inputs()
        self._build_layers()
    
    def _validate_inputs(self) -> None:
        assert self.p_dropout >= 0 and self.p_dropout < 1, "Dropout probability must be in [0, 1)"
        assert self.kernel_size <= min(self.input_width, self.input_height), "Kernel size must be smaller than input size"
        assert self.activation in ACTIVATIONS, "Activation function must be one of {}".format(ACTIVATIONS.keys())
    
    def _build_layers(self) -> None:
        """
        Build the convolutional layers. The stride will be decremented by 1 for each layer, and
        the kernel size will be decremented by 2 for each layer. 
        """
        conv_layers, dense_layers = [], []
        kernel_size, stride = self.kernel_size, self.stride
        input_width, input_height = self.input_width, self.input_height
        while kernel_size < min(input_width, input_height):
            if not conv_layers:
                conv_layers.append(nn.Conv2d(self.num_channels, self.intermediate_channels, kernel_size, stride))
            else:
                conv_layers.append(nn.Conv2d(self.intermediate_channels, self.intermediate_channels, kernel_size, stride))
            if self.batch_norm:
                conv_layers.append(nn.BatchNorm2d(self.intermediate_channels))
            conv_layers.append(ACTIVATIONS[self.activation])
            if self.p_dropout > 0:
                conv_layers.append(nn.Dropout2d(self.p_dropout))
            input_width = (input_width - kernel_size) // stride + 1
            input_height = (input_height - kernel_size) // stride + 1
            if kernel_size > 3:
                kernel_size -= 2
            if stride > 2:
                stride -= 1
        conv_layers.append(nn.Flatten())
        dense_layers.append(nn.Linear(input_width * input_height * self.intermediate_channels, self.dense_hidden_dim))
        dense_layers.append(ACTIVATIONS[self.activation])
        if self.p_dropout > 0:
            dense_layers.append(nn.Dropout(self.p_dropout))
        dense_layers.append(nn.Linear(self.dense_hidden_dim, 1))
        self.conv_layers = nn.Sequential(*conv_layers)
        self.dense_layers = nn.Sequential(*dense_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_layers(self.conv_layers(x))


class DeepKrigingMLP(MLP):
    """
    Defines a DeepKriging model with MLP as the backbone

    Arguments
    --------------
    input_dim: int, the dimension of input tensor
    num_hidden_layers: int, the number of hidden layers
    hidden_dims: int or list of int, the dimensions of hidden layers
    K: int, the basis size, default to 4
    batch_norm: bool, whether to use batch normalization in each hidden layer, default to False
    p_dropout: float, the dropout probability, default to 0.0
    activation: str, the activation function to use, default to "relu"
    """
    def __init__(self, input_dim: int, num_hidden_layers: int, hidden_dims: Union[int, List[int]], K: int = 1,
                 batch_norm: bool = False, p_dropout: float = 0.0, activation: str = "relu") -> None:
        self.K: int = K
        super(DeepKrigingMLP, self).__init__(input_dim, num_hidden_layers, hidden_dims, batch_norm, 
                                             p_dropout, activation)
        self._build_layers()
    
    def _build_layers(self) -> None:
        super(DeepKrigingMLP, self)._build_layers()
        self.embedding = DeepKrigingEmbedding2D(self.K)
        out_feature = self.hidden_dims[0] if isinstance(self.hidden_dims, List) else self.hidden_dims
        self.mlp_layers[0] = nn.Linear(
            self.mlp_layers[0].in_features + sum(self.embedding.num_basis), out_feature
        )
    
    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        x: torch.Tensor, shape (N, D), the input tensor
        s: torch.Tensor, shape (N,) or (N,1), the spatial information
        """
        phi = self.embedding(s)
        return self.mlp_layers(torch.cat([x, phi], dim=1))


class DeepKrigingConvNet(ConvNet):
    """
    Defines a DeepKriging model with CNN as the backbone

    Arguments
    --------------
    input_width: int, the width of input tensor
    input_height: int, the height of input tensor
    K: int, the basis size, default to 4
    num_channels: int, number of input channels, default to 1
    kernel_size: int, the initial kernel size of each convolutional layer, default to 7
    stride: int, the initial stride of each convolutional layer, default to 3
    intermediate_channels: int, the maximum number of channels in the convolutional layers, default to 64
    batch_norm: bool, whether to use batch normalization in each convolutional block, default to False
    p_dropout: float, the dropout probability, default to 0.0
    activation: str, the activation function to use, default to "tanh"
    """
    def __init__(self, input_width: int, input_height: int, K: int = 1, num_channels: int = 1, kernel_size: int = 7, 
                 stride: int = 3, intermediate_channels: int = 64, dense_hidden_dim: int = 128, batch_norm: bool = False, 
                 p_dropout: float = 0.0, activation: str = "relu") -> None:
        self.K: int = K
        super(DeepKrigingConvNet, self).__init__(input_width, input_height, num_channels, kernel_size, stride, 
                                                 intermediate_channels, dense_hidden_dim, batch_norm, p_dropout, 
                                                 activation)
        self._build_layers()
    
    def _build_layers(self) -> None:
        super(DeepKrigingConvNet, self)._build_layers()
        self.embedding = DeepKrigingEmbedding2D(self.K)
        self.dense_layers[0] = nn.Linear(
            self.dense_layers[0].in_features + sum(self.embedding.num_basis), self.dense_hidden_dim
        )
    
    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        x: torch.Tensor, shape (N, C, H, W), the input tensor
        s: torch.Tensor, shape (N,) or (N,1), the spatial information
        """
        phi = self.embedding(s)
        return self.dense_layers(torch.cat([self.conv_layers(x), phi], dim=1))


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
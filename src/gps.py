# gps.py: a file containing definitions for the Generalized Propensity Score Model
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import torch
from torch import nn
from nn import MLP
from typing import List, Union
from scipy.stats import norm

class GeneralizedPropensityScoreModel(MLP):
    """
    Generalized Propensity Score Model

    Arguments
    --------------
    input_dim: int, The dimension of the covariate X + spatial information s
    """
    def __init__(self, input_dim: int, num_hidden_layers: int, hidden_dims: Union[int, List[int]], 
                 batch_norm: bool = False, p_dropout: float = 0.0, activation: str = "relu") -> None:
        super(GeneralizedPropensityScoreModel, self).__init__(input_dim, num_hidden_layers, 
            hidden_dims,batch_norm, p_dropout, activation)
        if num_hidden_layers > 0:
            last_hidden_dim = hidden_dims[-1] if isinstance(hidden_dims, List) else hidden_dims
        else:
            last_hidden_dim = input_dim
        self.mlp_layers[-1] = nn.Linear(last_hidden_dim, 2)
    
    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        output = self.mlp_layers(torch.cat([x, s], dim=1))
        mean = output[:, 0]
        var = torch.exp(output[:, 1])
        return mean, var
    
    def generate_propensity_score(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mean, var = self.forward(x, s)
            return norm.pdf(t, mean, torch.sqrt(var))

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
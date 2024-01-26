# models.py: a file containing all models for nonlinear spatial causal inference
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import torch
from torch import nn

from typing import List
from nn import MLP, ConvNet, DeepKrigingMLP, DeepKrigingConvNet
from ick.kernels.kernel_fn import *
from ick.model.ick import ICK

KERNEL_FUNCS = {
    'rbf': sq_exp_kernel_nys, 
    'periodic': periodic_kernel_nys, 
    'exp': exp_kernel_nys,
    'rq': rq_kernel_nys, 
    'matern_type1': matern_type1_kernel_nys,
    'matern_type2': matern_type2_kernel_nys, 
    'linear': linear_kernel_nys,
    'spectral_mixture': spectral_mixture_kernel_1d_nys
}

KERNEL_PARAMS = {
    'rbf': ['std','lengthscale','noise'], 
    'periodic': ['std','period','lengthscale','noise'], 
    'exp': ['std','lengthscale','noise'],
    'rq': ['std','lengthscale','scale_mixture','noise'], 
    'matern_type1': ['std','lengthscale','noise'],
    'matern_type2': ['std','lengthscale','noise'], 
    'linear': ['std','c','noise'],
    'spectral_mixture': ['weight','mean','cov','noise']
}

class NonlinearSCI(nn.Module):
    """
    Defines the model for nonlinear spatial causal inference
    y = beta_1 * T1 + ... + beta_m + Tm + f_1(T1_bar) + ... + f_m(Tm_bar) + g(X) + U + epsilon
    where T is the intervention variable, T_bar is the neighboring interventions, X is the confounder,
    U is the unobserved confounder, epsilon is the noise term, and m is the number of interventions.
    Here both f and g are deep models, and U is modeled using the Implicit Composite Kernel (ICK).

    Arguments
    --------------
    num_interventions: int, number of interventions
    window_size: int, grid size for neighboring interventions T_bar
    confounder_dim: int, dimension of the confounder X
    f_network_type: str, the model type for f, default to "convnet"
    g_network_type: str, the model type for g, default to "mlp"
    unobserved_confounder: bool, whether to include the unobserved confounder U, default to False
    **kwargs: dict, additional keyword arguments for f, g, and U
        arguments for f:
        - f_channels: int, number of channels for the convolutional neural network for f, default to 1
        - f_num_basis: int, number of basis functions for the deep kriging model for f, default to 4
        arguments for g:
        - g_hidden_dims: List[int], the dimensions of hidden layers for the MLP for g, default to [128,64]
        - g_channels: int, number of channels for the convolutional neural network for g, default to 1
        - g_num_basis: int, number of basis functions for the deep kriging model for g, default to 4
        arguments for U:
        - kernel_func: str, the kernel function to use for the ICK model
        - kernel_param_vals: List[float], the initial values for the kernel parameters
        - inducing_point_space: List[List[float]], the space of the inducing points for Nystrom approximation
    """
    def __init__(self, num_interventions: int, window_size: int, confounder_dim: int, f_network_type: str = "convnet", 
                 g_network_type: str = "mlp", unobserved_confounder: bool = False, **kwargs) -> None:
        super(NonlinearSCI, self).__init__()
        self.num_interventions: int = num_interventions
        self.window_size: int = window_size
        self.confounder_dim: int = confounder_dim
        self.f_network_type: str = f_network_type
        self.g_network_type: str = g_network_type
        self.unobserved_confounder: bool = unobserved_confounder
        self.kwargs: dict = kwargs
        self._build_model()
    
    def _build_model(self) -> None:
        """
        Build the nonlinear spatial causal inference model, where the intervention variables are modeled
        using a linear relationship, the spatial/non-spatial confounders are modeled using convolutional 
        neural networks, and the unobserved confounder is modeled using a pseudo Gaussian process, which
        is implemented using the Implicit Composite Kernel (ICK) model.

        References:
        Jiang, Ziyang, et al. "Incorporating prior knowledge into neural networks through an implicit 
        composite kernel." arXiv preprint arXiv:2205.07384 (2022).
        """
        for i in range(1,self.num_interventions+1):
            setattr(self, f"beta_{i}", nn.Parameter(torch.randn(1)))
        # model for f(T_bar)
        if self.f_network_type == "convnet":
            f_channels = self.kwargs.get('f_channels', 1)
            for i in range(1,self.num_interventions+1):
                setattr(self, f"f_{i}", ConvNet(self.window_size, self.window_size, f_channels))
        elif self.f_network_type == "dk_convnet":
            f_channels = self.kwargs.get('f_channels', 1)
            f_num_basis = self.kwargs.get('f_num_basis', 4)
            for i in range(1,self.num_interventions+1):
                setattr(self, f"f_{i}", DeepKrigingConvNet(self.window_size, self.window_size, f_num_basis, f_channels))
        else:
            raise Exception(f"Invalid network type for f: {self.f_network_type}")
        # model for g(X)
        if self.g_network_type == "mlp":
            g_hidden_dims = self.kwargs.get('g_hidden_dims', [128,64])
            self.g = MLP(self.confounder_dim, len(g_hidden_dims), g_hidden_dims)
        elif self.g_network_type == "convnet":
            g_channels = self.kwargs.get('g_channels', 1)
            self.g = ConvNet(self.window_size, self.window_size, g_channels)
        elif self.g_network_type == "dk_mlp":
            g_hidden_dims = self.kwargs.get('g_hidden_dims', [128,64])
            g_num_basis = self.kwargs.get('g_num_basis', 4)
            self.g = DeepKrigingMLP(self.confounder_dim, len(g_hidden_dims), g_hidden_dims, g_num_basis)
        elif self.g_network_type == "dk_convnet":
            g_channels = self.kwargs.get('g_channels', 1)
            g_num_basis = self.kwargs.get('g_num_basis', 4)
            self.g = DeepKrigingConvNet(self.window_size, self.window_size, g_num_basis, g_channels)
        else:
            raise Exception(f"Invalid network type for g: {self.g_network_type}")
        # model for U
        if self.unobserved_confounder:
            kernel_assignment = ['ImplicitNystromKernel']
            kernel_params = {
                'ImplicitNystromKernel': {
                    'kernel_func': KERNEL_FUNCS[self.kwargs.get('kernel_func', 'rbf')], 
                    'params': KERNEL_PARAMS[self.kwargs.get('kernel_func', 'rbf')], 
                    'vals': self.kwargs.get('kernel_param_vals', [1.,1.,0.5]), 
                    'trainable': [True] * len(KERNEL_PARAMS[self.kwargs.get('kernel_func', 'rbf')]), 
                    'alpha': self.kwargs.get('alpha', 1e-5),
                    'num_inducing_points': self.kwargs.get('num_inducing_points', 32),
                    'nys_space': self.kwargs.get('inducing_point_space', [[0.,1.]])
                }
            }
            self.gp_unobserved_confounder = ICK(kernel_assignment, kernel_params)
    
    def forward(self, t: List[torch.Tensor], x: torch.Tensor, s: torch.Tensor = None) -> torch.Tensor:
        """
        t: List[torch.Tensor], a list of tensors containing the intervention variables with shape 
            (batch_size, window_size, window_size)
        x: torch.Tensor, a tensor containing the confounder variable with shape (batch_size, confounder_dim) or
            (batch_size, window_size, window_size)
        s: torch.Tensor, a tensor containing the spatial information (e.g., coordinates or distance) 
            of the training data, only used when unobserved_confounder is True
        """
        y_t, y_t_bar, y_x = [], [], []
        t_mask = torch.ones_like(t[0])
        t_mask[:,self.window_size//2,self.window_size//2] = 0
        for i in range(self.num_interventions):
            ti = t[i][:,self.window_size//2,self.window_size//2]
            y_t.append(ti * getattr(self, f"beta_{i+1}"))
            ti_bar = (t[i] * t_mask).unsqueeze(1)     # broadcasting
            if self.f_network_type == "convnet":
                y_t_bar.append(getattr(self, f"f_{i+1}")(ti_bar).squeeze())
            elif self.f_network_type == "dk_convnet":
                y_t_bar.append(getattr(self, f"f_{i+1}")(ti_bar, s).squeeze())
        if self.g_network_type == "mlp":
            y_x.append(self.g(x).squeeze())
        elif self.g_network_type == "convnet":
            y_x.append(self.g(x.unsqueeze(1)).squeeze())
        elif self.g_network_type == "dk_mlp":
            y_x.append(self.g(x, s).squeeze())
        elif self.g_network_type == "dk_convnet":
            y_x.append(self.g(x.unsqueeze(1), s).squeeze())
        output = torch.sum(torch.stack(y_t + y_t_bar + y_x), dim=0)
        if len(s.shape) == 1:
            s = s.unsqueeze(1)
        return output if not self.unobserved_confounder else (output + self.gp_unobserved_confounder([s]))
        
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
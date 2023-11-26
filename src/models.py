# models.py: a file containing all models for nonlinear spatial causal inference
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import torch
from torch import nn

from typing import List
from networks import ConvNet
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

    Arguments
    --------------
    num_interventions: int, the total number of treatments
    num_confounders: int, the total number of non-spatial confounders
    num_spatial_confounders: int, the total number of spatial confounders
    window_size: int, grid size for spatial/non-spatial confounders
    confounder_channels: int, number of channels for non-spatial confounders, default to 3
    spatial_confounder_channels: int, number of channels for spatial confounders, default to 1
    unobserved_confounder: bool, whether to include an unobserved confounder
    intervention_coeffs: List[int], the coefficients for each intervention, optional
        if not provided, the coefficients will be randomly initialized from N(0,1)
    **kwargs: dict, additional keyword arguments for the Implicit Composite Kernel (ICK) model
        - kernel_func: str, the kernel function to use for the ICK model
        - kernel_param_vals: List[float], the initial values for the kernel parameters
        - inducing_point_space: List[List[float]], the space of the inducing points for Nystrom approximation
    """
    def __init__(self, num_interventions: int, num_confounders: int, num_spatial_confounders: int, 
                 window_size: int, confounder_channels: int = 3, spatial_confounder_channels: int = 1, 
                 unobserved_confounder: bool = False, intervention_coeffs: List[float] = None, **kwargs) -> None:
        super(NonlinearSCI, self).__init__()
        self.num_interventions: int = num_interventions
        self.num_confounders: int = num_confounders
        self.num_spatial_confounders: int = num_spatial_confounders
        self.window_size: int = window_size
        self.confounder_channels: int = confounder_channels
        self.spatial_confounder_channels: int = spatial_confounder_channels
        self.unobserved_confounder: bool = unobserved_confounder
        self.intervention_coeffs: List[float] = intervention_coeffs
        self.kwargs: dict = kwargs
        self._validate_inputs()
        self._build_model()
    
    def _validate_inputs(self) -> None:
        if self.intervention_coeffs is not None:
            assert len(self.intervention_coeffs) == self.num_interventions, \
                "Number of intervention coefficients must match number of interventions."
        if self.unobserved_confounder:
            assert 'kernel_func' in self.kwargs.keys(), \
                "Must specify a kernel function for the ICK model."
            assert self.kwargs['kernel_func'] in KERNEL_FUNCS.keys(), \
                "The kernel function must be one of the following: rbf, periodic, exp, rq, \
                matern_type1, matern_type2, linear, spectral_mixture."
            assert 'kernel_param_vals' in self.kwargs.keys(), \
                "Must provide initial values for kernel parameters."
            assert len(self.kwargs['kernel_param_vals']) == len(KERNEL_PARAMS[self.kwargs['kernel_func']]), \
                "Number of initial values for kernel parameters must match number of kernel parameters."
            assert 'inducing_point_space' in self.kwargs.keys(), \
                "Must provide space of inducing points for Nystrom approximation."
    
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
        for i in range(self.num_interventions):
            if self.intervention_coeffs:
                setattr(self, f"coeff_{i}", nn.Parameter(torch.tensor(self.intervention_coeffs[i])))
            else:
                setattr(self, f"coeff_{i}", nn.Parameter(torch.randn(1)))
        for i in range(self.num_confounders):
            setattr(self, f"convnet_confounder_{i}", ConvNet(self.window_size, self.window_size, self.confounder_channels))
        for i in range(self.num_spatial_confounders):
            setattr(self, f"convnet_spatial_confounder_{i}", ConvNet(self.window_size, self.window_size, self.spatial_confounder_channels))
        if self.unobserved_confounder:
            kernel_assignment = ['ImplicitNystromKernel']
            kernel_params = {
                'ImplicitNystromKernel': {
                    'kernel_func': KERNEL_FUNCS[self.kwargs['kernel_func']], 
                    'params': KERNEL_PARAMS[self.kwargs['kernel_func']], 
                    'vals': self.kwargs['kernel_param_vals'], 
                    'trainable': [True] * len(KERNEL_PARAMS[self.kwargs['kernel_func']]), 
                    'alpha': self.kwargs.get('alpha', 1e-5),
                    'num_inducing_points': self.kwargs.get('num_inducing_points', 32),
                    'nys_space': self.kwargs['inducing_point_space']
                }
            }
            self.gp_unobserved_confounder = ICK(kernel_assignment, kernel_params)
        
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
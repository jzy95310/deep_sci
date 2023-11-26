# models.py: a file containing all models for nonlinear spatial causal inference
# SEE LICENSE STATEMENT AT THE END OF THE FILE

import torch
from torch import nn

from typing import List
from networks import ConvNet

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
    """
    def __init__(self, num_interventions: int, num_confounders: int, num_spatial_confounders: int, 
                 window_size: int, confounder_channels: int = 3, spatial_confounder_channels: int = 1, 
                 unobserved_confounder: bool = False, intervention_coeffs: List[float] = None) -> None:
        super(NonlinearSCI, self).__init__()
        self.num_interventions: int = num_interventions
        self.num_confounders: int = num_confounders
        self.num_spatial_confounders: int = num_spatial_confounders
        self.window_size: int = window_size
        self.confounder_channels: int = confounder_channels
        self.spatial_confounder_channels: int = spatial_confounder_channels
        self.unobserved_confounder: bool = unobserved_confounder
        self.intervention_coeffs: List[float] = intervention_coeffs
        self._validate_inputs()
        self._build_model()
    
    def _validate_inputs(self) -> None:
        if self.intervention_coeffs is not None:
            assert len(self.intervention_coeffs) == self.num_interventions, \
                "Number of intervention coefficients must match number of interventions."
    
    def _build_model(self) -> None:
        for i in range(self.num_interventions):
            if self.intervention_coeffs:
                setattr(self, f"coeff_{i}", nn.Parameter(torch.tensor(self.intervention_coeffs[i])))
            else:
                setattr(self, f"coeff_{i}", nn.Parameter(torch.randn(1)))
        for i in range(self.num_confounders):
            setattr(self, f"convnet_confounder_{i}", ConvNet(self.window_size, self.window_size, self.confounder_channels))
        for i in range(self.num_spatial_confounders):
            setattr(self, f"convnet_spatial_confounder_{i}", ConvNet(self.window_size, self.window_size, self.spatial_confounder_channels))
        
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
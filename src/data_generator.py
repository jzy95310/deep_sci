# data_generator.py: a file containing all classes of dataset definition
# SEE LICENSE STATEMENT AT THE END OF THE FILE

from typing import List, Tuple
import numpy as np

import torch
from torch.utils.data import Dataset

class SpatialDataset(Dataset):
    """
    Defines a dataset containing all variables for spatial causal inference

    Arguments
    --------------
    features: List[np.ndarray], A list of numpy arrays containing the gridded features for each sample
    spatial_features: np.ndarray, A numpy array containing the spatial features for each sample
    targets: List, A list of the targets for each sample
    """
    def __init__(self, features: List[np.ndarray], spatial_features: np.ndarray, targets: np.ndarray) -> None:
        self.features: List[np.ndarray] = features
        self.spatial_features: np.ndarray = spatial_features
        self.targets: np.ndarray = targets
        self._validate_and_preprocess_inputs()
        
    def _validate_and_preprocess_inputs(self) -> None:
        assert all([len(feat) == len(self.targets) for feat in self.features]), \
            "Features and targets must be the same length"
        assert len(self.spatial_features) == len(self.targets), \
            "Spatial features and targets must be the same length"
    
    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, idx) -> Tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return list([torch.from_numpy(feat[idx]).float() for feat in self.features]), \
            torch.from_numpy(self.spatial_features[idx]).float(), torch.from_numpy(self.targets[idx]).float()

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
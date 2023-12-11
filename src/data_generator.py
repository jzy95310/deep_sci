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
    def __init__(self, features: List, spatial_features: np.ndarray, targets: np.ndarray) -> None:
        self.features: List = features
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
        return list([torch.from_numpy(feat[idx]).float() if isinstance(feat[idx], np.ndarray) else torch.tensor(feat[idx]).float() for feat in self.features]), \
            torch.from_numpy(self.spatial_features[idx]).float() if isinstance(self.spatial_features[idx], np.ndarray) else torch.tensor(self.spatial_features[idx]), \
            torch.tensor(self.targets[idx]).float()

def train_val_test_split(features: List, spatial_features: np.ndarray, targets: np.ndarray, train_size: float = 0.7,
                         val_size: float = 0.15, test_size: float = 0.15, shuffle: bool = True, random_state: int = 2020) -> Tuple:
    """
    Splits the dataset into training, validation, and test sets
    If shuffle is set to True, the data will be shuffled before splitting
    """
    assert train_size + val_size + test_size == 1.0, "Train, validation, and test sizes must sum to 1"
    assert all([len(feat) == len(targets) for feat in features]), "Features and targets must be the same length"
    assert len(spatial_features) == len(targets), "Spatial features and targets must be the same length"

    if shuffle:
        np.random.seed(random_state)
        idx = np.random.permutation(len(targets))
        features = [feat[idx] for feat in features]
        spatial_features = spatial_features[idx]
        targets = targets[idx]
    
    train_idx = int(train_size * len(targets))
    val_idx = int((train_size + val_size) * len(targets))

    train_features, train_spatial_features = [feat[:train_idx] for feat in features], spatial_features[:train_idx]
    val_features, val_spatial_features = [feat[train_idx:val_idx] for feat in features], spatial_features[train_idx:val_idx]
    test_features, test_spatial_features = [feat[val_idx:] for feat in features], spatial_features[val_idx:]
    train_targets, val_targets, test_targets = targets[:train_idx], targets[train_idx:val_idx], targets[val_idx:]

    return SpatialDataset(train_features, train_spatial_features, train_targets), \
        SpatialDataset(val_features, val_spatial_features, val_targets), \
        SpatialDataset(test_features, test_spatial_features, test_targets)

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
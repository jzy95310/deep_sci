# data_generator.py: a file containing all classes of dataset definition
# SEE LICENSE STATEMENT AT THE END OF THE FILE

from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans

import torch
from torch.utils.data import Dataset

class SpatialDataset(Dataset):
    """
    Defines a dataset containing all variables for spatial causal inference

    Arguments
    --------------
    t: List[np.ndarray], A list of numpy arrays containing the interventions for each sample
    x: np.ndarray, A numpy array containing the confounder for each sample
    s: np.ndarray, A numpy array containing the spatial information for each sample
    y: List, A list of the targets for each sample
    """
    def __init__(self, t: List, x: np.ndarray, s: np.ndarray, y: np.ndarray) -> None:
        self.t: List = t
        self.x: np.ndarray = x
        self.s: np.ndarray = s
        self.y: np.ndarray = y
        self._validate_and_preprocess_inputs()
        
    def _validate_and_preprocess_inputs(self) -> None:
        assert all([len(feat) == len(self.y) for feat in self.t]), \
            "Interventions and targets must be the same length"
        assert len(self.x) == len(self.y), \
            "Confounder and targets must be the same length"
        assert len(self.s) == len(self.y), \
            "Spatial features and targets must be the same length"
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx) -> Tuple:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        batch = list([torch.from_numpy(feat[idx]).float() if isinstance(feat[idx], np.ndarray) else torch.tensor(feat[idx]).float() for feat in self.t]), \
            torch.from_numpy(self.x[idx]).float() if isinstance(self.x[idx], np.ndarray) else torch.tensor(self.x[idx]), \
            torch.from_numpy(self.s[idx]).float() if isinstance(self.s[idx], np.ndarray) else torch.tensor(self.s[idx]), \
            torch.from_numpy(self.y[idx]).float() if isinstance(self.y[idx], np.ndarray) else torch.tensor(self.y[idx])
        return batch

def train_val_test_split(t: List, x: np.ndarray, s: np.ndarray, y: np.ndarray, train_size: float = 0.7,
                         val_size: float = 0.15, test_size: float = 0.15, shuffle: bool = True, random_state: int = 2020, 
                         block_sampling: bool = False, num_blocks: int = 50, return_test_indices: bool = False) -> Tuple:
    """
    Splits the dataset into training, validation, and test sets
    If shuffle is set to True, the data will be shuffled before splitting
    If block_sampling is set to True, the data will first be clustered into the specified number of blocks using K-means
    and then splitted into training, validation, and test sets by sampling from the blocks
    """
    assert train_size + val_size + test_size == 1.0, "Train, validation, and test sizes must sum to 1"
    assert all([len(feat) == len(y) for feat in t]), "Interventions and targets must be the same length"
    assert len(x) == len(y), "Confounder and targets must be the same length"
    assert len(s) == len(y), "Spatial features and targets must be the same length"

    if shuffle:
        np.random.seed(random_state)
        idx = np.random.permutation(len(y))
        t = [feat[idx] for feat in t]
        x, s, y = x[idx], s[idx], y[idx]
    
    if block_sampling:
        kmeans = KMeans(n_clusters=num_blocks, random_state=random_state).fit(s)
        block_labels = kmeans.labels_
        block_indices = np.arange(num_blocks)
        train_blocks = np.random.choice(block_indices, size=int(train_size * num_blocks), replace=False)
        block_indices = np.setdiff1d(block_indices, train_blocks)
        val_blocks = np.random.choice(block_indices, size=int(val_size * num_blocks), replace=False)
        test_blocks = np.setdiff1d(block_indices, val_blocks)
        train_idx = np.where(np.isin(block_labels, train_blocks))[0]
        val_idx = np.where(np.isin(block_labels, val_blocks))[0]
        test_idx = np.where(np.isin(block_labels, test_blocks))[0]

        t_train, x_train, s_train = [feat[train_idx] for feat in t], x[train_idx], s[train_idx]
        t_val, x_val, s_val = [feat[val_idx] for feat in t], x[val_idx], s[val_idx]
        t_test, x_test, s_test = [feat[test_idx] for feat in t], x[test_idx], s[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    else:
        train_idx = int(train_size * len(y))
        val_idx = int((train_size + val_size) * len(y))

        t_train, x_train, s_train = [feat[:train_idx] for feat in t], x[:train_idx], s[:train_idx]
        t_val, x_val, s_val = [feat[train_idx:val_idx] for feat in t], x[train_idx:val_idx], s[train_idx:val_idx]
        t_test, x_test, s_test = [feat[val_idx:] for feat in t], x[val_idx:], s[val_idx:]
        y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]

    res = tuple([
        SpatialDataset(t_train, x_train, s_train, y_train), 
        SpatialDataset(t_val, x_val, s_val, y_val),
        SpatialDataset(t_test, x_test, s_test, y_test)
    ])
    if return_test_indices:
        res += (test_idx,) if block_sampling else (idx[val_idx:],)
    return res

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
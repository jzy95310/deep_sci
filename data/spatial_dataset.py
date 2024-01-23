import os
import numpy as np
from scipy.signal import convolve2d

from sklearn.neural_network import MLPRegressor # for a random function generator
import rasterio as rio # To handle working with geospatial data
import multiprocessing as mp # To parallelize the computation of the NLCD weights

# For creating a smooth random function
from scipy.interpolate import CubicSpline

class SpatialDataset:
    def __init__(
        self,
        dataset_path,
        window_size,
        num_samples,
        len_scale,
        seed=42
    ):
        """
        This class is used to generate the dataset for spatial confounding

        Args:
          dataset_path: where the data is stored.
          windoow_size: the size of the window around the center to consider
          num_samples: the total number of samples to generate for the dataset
          seed: the random seed to use for the dataset
          
        """

        # Load NLCD, NDVI, and unobserved confounder
        with rio.open(os.path.join(dataset_path, "durham_nlcd.tif")) as src:
            nlcd = src.read(1)

        with rio.open(os.path.join(dataset_path, "durham_ndvi.tif")) as src:
            ndvi = src.read(1)
        ndvi = np.flipud(ndvi)

        with rio.open(os.path.join(dataset_path, "durham_synth_unobs_confound.tif")) as src:
            U = src.read(1)

        U = np.flipud(U)
            
        assert nlcd.shape == ndvi.shape
        assert nlcd.shape == U.shape

        # This line converts NLCD to a percentage representation
        # of the land cover class in the image
        self.nlcd = self._create_one_hot_encoding(nlcd)
        self.ndvi = ndvi
        self.U = U
        self.window_size = window_size

        # Generate a set of num_samples coords that are within the bounds of the image
        # and are not within window_size of the edge of the image
        np.random.seed(seed)
        self.coords = self._sample_coords(nlcd.shape, window_size, num_samples)

        # Reset the random seed
        np.random.seed(seed + 1)
        self.response = self._create_random_function(seed)

        self.wm = self._calc_weight_matrix(window_size, len_scale)

        # Lastly, generate the coefficients
        self.de_coef = 5
        self.confound_coef = np.random.uniform(0, 1, size=self.nlcd.shape[2])


    def _create_random_function(self, seed):
        x = np.linspace(0, 1, 10)

        y = np.random.multivariate_normal(3*x, 0.5*np.eye(10), 1)
        y = np.sort(y)
        y = y.reshape(-1, 1)

        # Now, we are going to create a cubic spline as the function
        cs = CubicSpline(x, y)

        return cs


    def _sample_coords(self, shape, window_size, num_samples):
        """This function samples num_samples coordinates from the image that are
        and those are the coords used for the dataset."""

        # Set the meshgrid to be the size of the image
        x = np.arange(1 + window_size, shape[0] - window_size)
        y = np.arange(1 + window_size, shape[1] - window_size)
        x, y = np.meshgrid(x, y)
        # Stack the coords so they can be sampled
        coords = np.vstack((x.flatten(), y.flatten())).T

        # Sample num_samples coords from the meshgrid, with
        # no replacement
        samples = np.random.choice(coords.shape[0], num_samples, replace=False)
        coords = coords[samples, :]
        return coords

    def calc_temp(self, ndvi_window, nlcd, U):
        """
        This function uses NDVI, NLCD, and U to return a temperature
        value for the given window.
        
        Arguments:
            ndvi_window: the window of NDVI values
            nlcd_window: the window of NLCD values
            U: the unobserved confounder

        """

        # Get the center of the ndvi_window
        ndvi_center = ndvi_window[self.window_size, self.window_size]

        # Run the ndvi_window through the random function
        # TODO: figure out what function to apply to the ndvi window
        indirect_effect = self.response(
            np.sum(self.wm * ndvi_window)
        )[0]

        # Return the temperature value
        return self.de_coef*ndvi_center + indirect_effect + np.dot(self.confound_coef, nlcd) + U

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):

        # Get the coordinates
        i, j = self.coords[idx]

        # Return the NLCD
        nlcd = self.nlcd[
            i - self.window_size : i + self.window_size + 1,
            j - self.window_size : j + self.window_size + 1,
            :
        ].mean(axis=(0,1))
        
        # Return the NDVI 
        ndvi = self.ndvi[
            i - self.window_size : i + self.window_size + 1,
            j - self.window_size : j + self.window_size + 1,
        ]

        # Looking at U
        U = self.U[i, j]

        temp = self.calc_temp(ndvi, nlcd, U)
        
        # Notably, we won't return U, since it is unobserved
        return i, j, nlcd, ndvi, temp

    def _create_one_hot_encoding(self, nlcd):
        # Creating the spatially aggregated one-hot encoding
        cat = len(np.unique(nlcd))
        # Initialize the one-hot encoding vector as zeros.
        one_hot = np.zeros((nlcd.shape[0], nlcd.shape[1], cat))

        # Set the values of the one-hot encoding based on the land use data.
        for i, segment in enumerate(np.unique(nlcd)):
            one_hot[:, :, i] = (nlcd == segment).astype(int)

        return one_hot

    def _task(self, args):
        weight = convolve2d(args[0], args[1], mode="same")
        # weight = (weight - np.mean(weight)) / (np.std(weight) + 1e-8)
        return weight

    def _calc_weight_matrix(self, window_size, length_scale):
        dist_matrix = np.sqrt(
            np.arange(-window_size, window_size + 1)[np.newaxis, :] ** 2
            + np.arange(-window_size, window_size + 1)[:, np.newaxis] ** 2
        )
        weight_matrix = np.exp(-dist_matrix / length_scale)
        weight_matrix /= weight_matrix.sum()
        return weight_matrix

    def _calc_nlcd_percentage(self, nlcd, window_size):
        one_hot = self._create_one_hot_encoding(nlcd)

        weight_matrix = np.ones((window_size * 2 + 1, window_size * 2 + 1))
        weight_matrix /= weight_matrix.sum()

        # Define the iterable
        iterable = [(one_hot[:, :, i], weight_matrix) for i in range(one_hot.shape[2])]

        nlcd_weight = np.zeros_like(one_hot)
        with mp.Pool(8) as pool:
            for i, result in enumerate(pool.map(self._task, iterable)):
                nlcd_weight[:, :, i] = result

        return nlcd_weight
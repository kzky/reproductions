import argparse
from nnabla.contrib.context import extension_context
from nnabla.monitor import Monitor, MonitorImage, MonitorImageTile, MonitorSeries, tile_images
from nnabla.utils.data_iterator import data_iterator
import os

import nnabla as nn
import nnabla.functions as F
import nnabla.logger as logger
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np


class MonitorImageTileWithName(MonitorImageTile):

    def __init__(self, name, monitor, num_images=4, normalize_method=None):
        super(MonitorImageTileWithName, self).__init__(
            name, monitor, interval=1000, verbose=True, num_images=num_images, normalize_method=normalize_method)

    def add(self, name, var):
        import nnabla as nn
        from scipy.misc import imsave
        if isinstance(var, nn.Variable):
            data = var.d.copy()
        elif isinstance(var, nn.NdArray):
            data = var.data.copy()
        else:
            assert isinstance(var, np.ndarray)
            data = var.copy()
        assert data.ndim > 2
        channels = data.shape[-3]
        data = data.reshape(-1, *data.shape[-3:])
        data = data[:min(data.shape[0], self.num_images)]
        data = self.normalize_method(data)
        if channels > 3:
            data = data[:, :3]
        elif channels == 2:
            data = np.concatenate(
                [data, np.ones((data.shape[0], 1) + data.shape[-2:])], axis=1)
        tile = tile_images(data)
        path = os.path.join(self.save_dir, '{}.png'.format(name))
        imsave(path, tile)

def generate_gaussian_noise(shape, noise_level):
    size = np.prod(shape)
    stds = np.random.choice(np.arange(noise_level), size=size, replace=True)
    noise = np.random.normal(0, stds, size).reshape(shape)
    return noise


def generate_possion_noise(shape, noise_level):
    size = np.prod(shape)
    lambda_ = np.random.choice(np.arange(noise_level), size=size, replace=True)
    noise = np.random.poisson(lambda_, size).reshape(shape)
    return noise


def generate_possion_bernoulli(shape, noise_level):
    size = np.prod(shape)
    noise = np.random.randint(2, size)
    return noise


def apply_noise(x, noise_level, distribution="gaussian"):
    if distribution == "gaussian":
        return x + generate_gaussian_noise(x.shape, noise_level)
    elif distribution == "poisson":
        return x + generate_poisson_noise(x.shape, noise_level)
    elif distribution == "bernoulli":
        return x * generate_bernoulli(x.shape, noise_level)
    else:
        raise ValueError("distribution = {} is not supported.".format(distribution))
    
    

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
            name, monitor, interval=10000, verbose=True, num_images=num_images, normalize_method=normalize_method)

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


def generate_gaussian_noise(shape, noise_level, fix=False):
    size = np.prod(shape)
    noise_level = int(noise_level)
    if fix:
        stds = np.asarray([noise_level] * size)
    else:
        stds = np.random.choice(np.arange(noise_level), size=size, replace=True)
    noise = np.random.normal(0, stds, size).reshape(shape)
    return noise


def generate_poisson_noise(shape, noise_level, fix=False):
    size = np.prod(shape)
    noise_level = int(noise_level)
    if fix:
        lambda_ = np.asarray([noise_level] * size)
    else:
        lambda_ = np.random.choice(np.arange(noise_level), size=size, replace=True)
    noise = np.random.poisson(lambda_, size).reshape(shape)
    return noise


def generate_bernoulli_noise(shape, noise_level=0.95, fix=False):
    size = np.prod(shape)
    p = np.random.uniform(0, noise_level, size=size)
    noise = np.random.binomial(1, p, size=size).reshape(shape)
    return noise


def generate_impulse_noise(shape, noise_level=0.95, fix=False):
    size = np.prod(shape)
    p = np.random.uniform(0, noise_level, size=size)
    m = np.random.binomial(1, p, size=size).reshape(shape)
    v = np.random.uniform(0, 256, size=size).reshape(shape)
    return m, v


def apply_noise(x, noise_level, distribution="gaussian", fix=False):
    if distribution == "gaussian":
        return x + generate_gaussian_noise(x.shape, noise_level, fix)
    elif distribution == "poisson":
        return x + generate_poisson_noise(x.shape, noise_level, fix)
    elif distribution == "bernoulli":
        return x * generate_bernoulli_noise(x.shape, noise_level, fix)
    elif distribution == "impulse":
        m, v = generate_impulse_noise(x.shape, noise_level, fix)
        #return x * m + v * (1 - m)
        return m * (x - v) + v
    elif distribution == "text":
        raise ValueError("distribution = {} is not supported.".format(distribution))
    else:
        raise ValueError("distribution = {} is not supported.".format(distribution))
    

def psnr(x, y, max_=255):
    mse = np.mean((x - y) ** 2)
    return 10 * np.log10(max_ ** 2 / mse)
    

class RampdownLRScheduler(object):

    def __init__(self, solver, interval=1000, decay_factor=0.5):
        """
        Args:
           solver (`S.solver`): solver
           interval (`int`): interval
           decay_fator (`int`): decay_fator
        """
        self.solver = solver
        self.interval = interval
        self.decay_factor = decay_factor
        self.losses = []

    def __call__(self, loss):
        """Set learning rate
        Args:
           loss (`numpy.ndarray`): loss

        """
        loss = loss.copy()
        if len(self.losses) < self.interval:
            self.losses.append(loss)
            return

        self.losses.pop(0)
        self.losses.append(loss)
        ave_loss = np.mean(self.losses)
        lr = self.solver.learning_rate()
        if loss > ave_loss:
            lr = lr * self.decay_factor
            logger.info("LR rampdowns to {}.".format(lr))
            self.losses = []
        self.solver.set_learning_rate(lr)

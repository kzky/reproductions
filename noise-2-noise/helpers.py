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


def generate_gaussian_noise(shape, noise_level, test=False):
    size = np.prod(shape)
    if test:
        std = noise_level
        noise = np.random.normal(0, std, size).reshape(shape)
    else:
        std = np.random.uniform(1, noise_level, size=size)
        noise = np.random.normal(0, std, size).reshape(shape)
    return noise


def generate_poisson_noise(shape, noise_level, test=False):
    """
    Noise is minus with lambda for adding minus noise
    """
    size = np.prod(shape)
    if test:
        lambda_ = noise_level
        noise = np.random.poisson(lambda_, size).reshape(shape)
        noise = noise - lambda_
    else:
        lambda_ = np.random.uniform(1, noise_level, size=size)
        noise = np.random.poisson(lambda_, size).reshape(shape)
        noise = noise - lambda_.reshape(shape)
    return noise


def generate_bernoulli_noise(shape, noise_level=0.95, test=False):
    size = np.prod(shape)
    if test:
        p = np.random.uniform(0, noise_level, size=1)
        noise = np.random.binomial(1, p, size=size).reshape(shape)
        noise = noise
    else:
        p = np.random.uniform(0, noise_level, size=size)
        noise = np.random.binomial(1, p, size=size).reshape(shape)
        noise = noise
        p = p.reshape(shape)
    return noise, p


def generate_impulse_noise(shape, noise_level=0.95, test=False):
    size = np.prod(shape)
    if test:
        p = np.random.uniform(0, noise_level, size=1)
        m = np.random.binomial(1, p, size=size).reshape(shape)
        v = np.random.choice(np.arange(255), size=size, replace=True).reshape(shape)
    else:
        p = np.random.uniform(0, noise_level, size=size)
        m = np.random.binomial(1, p, size=size).reshape(shape)
        v = np.random.choice(np.arange(255), size=size, replace=True).reshape(shape)
        p = p.reshape(shape)
    return m, v, p


def create_noisy_target(x_noise):
    # x_noise: (B, R, C, H, W)
    x_noisy_target = []
    for x in x_noise:
        target = np.broadcast_to(np.mean(x, axis=(0, )), x.shape)
        x_noisy_target.append(target)
    #return np.asarray(x_noisy_target)
    return x_noise - np.asarray(x_noisy_target)


def apply_noise(x, n_replica, noise_level, distribution="gaussian", test=False):
    # B, C, H, W
    shape = x.shape
    b, c, h, w = shape
    # B, R, C, H, W
    x = np.asarray(
        [np.broadcast_to(x_.reshape(1, c, h, w), (n_replica,) +  x.shape[1:]) for x_ in x])

    #TODO: change how to take the expectation according to loss?
    if distribution == "gaussian":
        n = generate_gaussian_noise(x.shape, noise_level, test)
        x_noise = x + n
        target = create_noisy_target(x_noise)
        x_noise, target = np.concatenate(x_noise), np.concatenate(target)
        return x_noise, target, None
    elif distribution == "poisson":
        n = generate_poisson_noise(x.shape, noise_level, test)
        x_noise = x + n
        target = create_noisy_target(x_noise)
        x_noise, target = np.concatenate(x_noise), np.concatenate(target)
        return x_noise, target, None
    elif distribution == "bernoulli":
        #TODO: wrong about how to create target noisy target
        n, p = generate_bernoulli_noise(x.shape, noise_level, test)
        x_noise = x * n
        #target = create_noisy_target(x_noise - x * p)
        #target = create_noisy_target(x_noise)
        target = create_noisy_target(x_noise - x_noise * p)
        x_noise, target, n = np.concatenate(x_noise), np.concatenate(target), np.concatenate(n)
        return x_noise, target, n
    elif distribution == "impulse":
        #TODO: how to create mask is wrong
        m, v, p = generate_impulse_noise(x.shape, noise_level, test)
        x_noise = (x - v) * m + v
        #target = create_noisy_target(x_noise - ((x - v) * p + v))
        target = create_noisy_target(x_noise)
        x_noise, target = np.concatenate(x_noise), np.concatenate(target)
        return x_noise, target, None
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
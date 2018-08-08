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
    if test:
        std = noise_level
        size = np.prod(shape)
        noise = np.random.normal(0, std, size).reshape(shape)
        noise = noise
        return noise
    else:
        b, r, c, h, w = shape
        noises = []
        stds = []
        for _ in range(b):  # Batch
            # assumption: noise level is the same for all replica (R x C x H x W)
            std = np.random.uniform(0, noise_level, size=1)
            noises_ = []
            for _ in range(r):  # Replica
                size = np.prod(c * h * w)
                noise = np.random.normal(0, std, size).reshape((c, h, w))
                noises_.append(noise)
            noises.append(np.asarray(noises_))
            stds.append(np.broadcast_to(std, (b, c, h, w)))
        return np.asarray(noises)

def generate_poisson_noise(shape, noise_level, test=False):
    """
    Noise is minus with lambda for adding minus noise
    """
    if test:
        lambda_ = int(noise_level)
        size = np.prod(shape)
        noise = np.random.poisson(lambda_, size).reshape(shape)
        noise = noise
        return noise, lambda_
    else:
        b, r, c, h, w = shape
        noises = []
        lambdas_ = []
        for _ in range(b):  # Batch
            # assumption: noise level is the same for all replica (R x C x H x W)
            lambda_ = np.random.uniform(0, int(noise_level), size=1)
            noises_ = []
            for _ in range(r):  # Replica
                size = np.prod(c * h * w)
                noise = np.random.poisson(lambda_, size).reshape((c, h, w))
                noises_.append(noise)
            noises.append(np.asarray(noises_))
            lambdas_.append(np.broadcast_to(lambda_, (r, c, h, w)))
        return np.asarray(noises), np.asarray(lambdas_)


def generate_bernoulli_noise(shape, noise_level=0.95, test=False):
    if test:
        p = np.random.uniform(0, noise_level, size=1)
        size = np.prod(shape)
        noise = np.random.binomial(1, p, size=size).reshape(shape)
        noise = noise
        return noise, p
    else:
        b, r, c, h, w = shape
        noises = []
        ps = []
        lambdas_ = []
        for _ in range(b):  # Batch
            # assumption: noise level is the same for all replica (R x C x H x W)
            noises_ = []
            p = np.random.uniform(0, noise_level, size=1)
            ps.append(p)
            for _ in range(r):  # Replica
                size = np.prod(h * w)  # use same mask for all channels
                noise = np.random.binomial(1, p, size=size).reshape((1, h, w))
                noises_.append(noise)
            noises.append(np.asarray(noises_))
        return np.asarray(noises), np.asarray(ps).reshape((b, 1, 1, 1, 1))


def generate_impulse_noise(shape, noise_level=0.95, test=False):
    if test:
        p = np.random.uniform(0, noise_level, size=1)
        size = np.prod(shape)
        m = np.random.binomial(1, p, size=size).reshape(shape)
        v = np.random.choice(np.arange(255), size=size, replace=True).reshape(shape)
        return m, v, p
    else:
        b, r, c, h, w = shape
        ms = []
        vs = []
        ps = []
        for _ in range(b):  # Batch
            # assumption: noise level is the same for all replica (R x C x H x W)
            noises_ = []
            ms_ = []
            vs_ = []
            p = np.random.uniform(0, noise_level, size=1)
            ps.append(p)            
            for _ in range(r):  # Replica
                size = np.prod(h * w)  # use same mask for all channels
                m = np.random.binomial(1, p, size=size).reshape((1, h, w))
                v = np.random.choice(np.arange(255), size=size, replace=True).reshape((1, h, w))
                ms_.append(m)
                vs_.append(v)
            ms.append(np.asarray(ms_))
            vs.append(np.asarray(vs_))
        return np.asarray(ms), np.asarray(vs), np.asarray(ps).reshape((b, 1, 1, 1, 1))


def mean_replicas(x_noise):
    # x_noise: (B, R, C, H, W)
    mean_replicas = []
    for x in x_noise:
        target = np.broadcast_to(np.mean(x, axis=(0, )), x.shape)
        mean_replicas.append(target)
    return np.asarray(mean_replicas)


def replicate_one_corrupted_sample(x_noisy_target):
    b, r, c, h, w = x_noisy_target.shape
    replicas = []
    for x in x_noisy_target:
        #x: (R, C, H, W)
        x = np.broadcast_to(x[0].reshape(1, c, h, w), (r, c, h, w))
        replicas.append(x)
    return np.asarray(replicas)
        

def apply_noise(x, n_replica, noise_level, distribution="gaussian", test=False):
    # B, C, H, W
    shape = x.shape
    b, c, h, w = shape
    r = n_replica
    # B, R, C, H, W
    x = np.asarray(
        [np.broadcast_to(x_.reshape(1, c, h, w), (n_replica,) +  x.shape[1:]) for x_ in x])

    #TODO: change how to take the expectation according to loss?
    if distribution == "gaussian":
        n = generate_gaussian_noise(x.shape, noise_level, test)
        x_noisy_target = np.clip(x + n, 0., 255.0)
        x_noise = replicate_one_corrupted_sample(x_noisy_target)
        shape = (b * r, c, h, w)
        return x_noise.reshape(shape), x_noisy_target.reshape(shape), None
    elif distribution == "poisson":
        n, lambda_ = generate_poisson_noise(x.shape, noise_level, test)
        x_noisy_target = np.clip(x + n - lambda_, 0., 255.0)
        x_noise = replicate_one_corrupted_sample(x_noisy_target)
        shape = (b * r, c, h, w)
        return x_noise.reshape(shape), x_noisy_target.reshape(shape), None
    elif distribution == "bernoulli":
        n, p = generate_bernoulli_noise(x.shape, noise_level, test)
        xn = x * n
        x_noisy_target = np.clip(xn - xn * p, 0., 255.0)
        x_noise = replicate_one_corrupted_sample(x_noisy_target)
        shape = (b * r, c, h, w)
        return x_noise.reshape(shape), x_noisy_target.reshape(shape), \
            np.broadcast_to(n, (b, r, c, h, w)).reshape(shape)
    elif distribution == "impulse":
        m, v, p = generate_impulse_noise(x.shape, noise_level, test)
        xn = v * m + x * (1 - m)
        x_noisy_target = np.clip(xn - xn * p, 0., 255.0)
        x_noise = replicate_one_corrupted_sample(x_noisy_target)
        shape = (b * r, c, h, w)
        return x_noise.reshape(shape), x_noisy_target.reshape(shape), None
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

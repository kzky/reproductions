import os
import numpy as np
import argparse
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.contrib.context import extension_context
from nnabla.monitor import Monitor, MonitorImageTile, MonitorSeries
from nnabla.parameter import get_parameter_or_create, get_parameter
from nnabla.initializer import (
    calc_uniform_lim_glorot,
    ConstantInitializer, NormalInitializer, UniformInitializer)
from nnabla.parametric_functions import parametric_function_api


@parametric_function_api("bn")
def BN(inp, axes=[1], decay_rate=0.9, eps=1e-5,
       batch_stat=True, output_stat=False, fix_parameters=False):
    """Batch Normalization
    """
    shape_stat = [1 for _ in inp.shape]
    shape_stat[axes[0]] = inp.shape[axes[0]]
    beta = get_parameter_or_create(
        "beta", shape_stat, ConstantInitializer(0), not fix_parameters)
    gamma = get_parameter_or_create(
        "gamma", shape_stat, ConstantInitializer(1), not fix_parameters)
    mean = get_parameter_or_create(
        "mean", shape_stat, ConstantInitializer(0), False)
    var = get_parameter_or_create(
        "var", shape_stat, ConstantInitializer(0), False)
    return F.batch_normalization(inp, beta, gamma, mean, var, axes,
                                 decay_rate, eps, batch_stat, output_stat)


@parametric_function_api("cbn")
def CBN(inp, z, axes=[1], decay_rate=0.9, eps=1e-5,
        batch_stat=True, output_stat=False, fix_parameters=False):
    """Conditional Batch Normalization
    """
    shape_stat = [1 for _ in inp.shape]
    shape_stat[axes[0]] = inp.shape[axes[0]]
    b, c, s0, s1 = inp.shape

    # Conditional normalization
    gamma = PF.affine(z, c, with_bias=False, name="gamma")
    gamma = F.reshape(gamma, shape_stat)
    beta = PF.affine(z, c, with_bias=False, name="beta")
    beta = F.reshape(beta, shape_stat)

    #TODO: have to use the composite BN
    raise Exception("Not implemented yet. Have to use use the composite BN!")

    # Batch normalization
    mean = get_parameter_or_create(
        "mean", shape_stat, ConstantInitializer(0), False)
    var = get_parameter_or_create(
        "var", shape_stat, ConstantInitializer(0), False)
    return F.batch_normalization(inp, beta, gamma, mean, var, axes,
                                 decay_rate, eps, batch_stat, output_stat)

@parametric_function_api("in")
def IN(inp, axes=[1], eps=1e-5, fix_parameters=False):
    """Instance Normalization
    """
    if inp.shape[0] == 1:
        return INByBatchNorm(inp, axes, 0.9, eps, fix_parameters)

    b, c = inp.shape[0:2]
    spacial_shape = inp.shape[2:]

    shape_stat = [1 for _ in inp.shape]
    shape_stat[axes[0]] = inp.shape[axes[0]]
    beta = get_parameter_or_create(
        "beta", shape_stat, ConstantInitializer(0), not fix_parameters)
    gamma = get_parameter_or_create(
        "gamma", shape_stat, ConstantInitializer(1), not fix_parameters)

    # Instance normalization
    axis = [i for i in range(len(inp.shape)) if i > 1]  # normalize over spatial dimensions
    mean = F.mean(inp, axis=axis, keepdims=True)
    var = F.mean(F.pow_scalar(inp - mean, 2.0), axis=axis, keepdims=True)
    h = (inp - mean) / F.pow_scalar(var + eps, 0.5)
    return gamma * inp + beta

@parametric_function_api("in")
def INByBatchNorm(inp, axes=[1], decay_rate=0.9, eps=1e-5, fix_parameters=False):
    """Instance Normalization (implemented using BatchNormalization)
    Instance normalization is equivalent to the batch normalization if a batch size is one, in
    other words, it normalizes over spatial dimension(s), meaning all dimensions except for
    the batch and feature dimension.
    """
    assert len(axes) == 1

    shape_stat = [1 for _ in inp.shape]
    shape_stat[axes[0]] = inp.shape[axes[0]]
    beta = get_parameter_or_create(
        "beta", shape_stat, ConstantInitializer(0), not fix_parameters)
    gamma = get_parameter_or_create(
        "gamma", shape_stat, ConstantInitializer(1), not fix_parameters)
    mean = get_parameter_or_create(
        "mean", shape_stat, ConstantInitializer(0), False)
    var = get_parameter_or_create(
        "var", shape_stat, ConstantInitializer(0), False)
    return F.batch_normalization(inp, beta, gamma, mean, var, axes,
                                 decay_rate, eps, True, False)

@parametric_function_api("cin")
def CIN(inp, z, eps=1e-5, fix_parameters=False):
    """Conditional Instance Normalization
    """
    if inp.shape[0] == 1:
        return CINByBatchNorm(inp, z, eps)

    b, c = inp.shape[0:2]
    spacial_shape = inp.shape[2:]

    # Conditional normalization
    gamma = PF.affine(z, c, with_bias=False, name="gamma")
    gamma = F.reshape(gamma, (b, c) + tuple([1 for _ in range(len(spacial_shape))]))
    gamma = F.broadcast(gamma, (b, c) + tuple([s for s in inp.shape[2:]]))
    beta = PF.affine(z, c, with_bias=False, name="beta")
    beta = F.reshape(beta, (b, c) + tuple([1 for _ in range(len(spacial_shape))]))
    beta = F.broadcast(beta, (b, c) + tuple([s for s in inp.shape[2:]]))

    # Instance normalization
    axis = [i for i in range(len(inp.shape)) if i > 1]  # normalize over spatial dimensions
    mean = F.mean(inp, axis=axis, keepdims=True)
    var = F.mean(F.pow_scalar(inp - mean, 2.0), axis=axis, keepdims=True)
    h = (inp - mean) / F.pow_scalar(var + eps, 0.5)
    
    return gamma * h + beta


def CINByBatchNorm(inp, z, eps=1e-5, axes=[1], fix_parameters=False):
    """Conditional Instance Normalization
    """
    # Batch size must be one.
    assert inp.shape[0] == z.shape[0]
    assert inp.shape[0] == 1
    assert z.shape[0] == 1

    b, c = inp.shape[0:2]
    spacial_shape = inp.shape[2:]

    # Conditional normalization
    gamma = PF.affine(z, c, with_bias=False, name="gamma")
    gamma = F.reshape(gamma, (b, c) + tuple([1 for _ in range(len(spacial_shape))]))
    gamma = F.broadcast(gamma, (b, c) + tuple([s for s in inp.shape[2:]]))
    beta = PF.affine(z, c, with_bias=False, name="beta")
    beta = F.reshape(beta, (b, c) + tuple([1 for _ in range(len(spacial_shape))]))
    beta = F.broadcast(beta, (b, c) + tuple([s for s in inp.shape[2:]]))

    # Instance normalization by using the batch normalization
    assert z.shape[0] == 1
    shape_stat = [1 for _ in inp.shape]
    shape_stat[axes[0]] = inp.shape[axes[0]]
    gamma_tmp = nn.Variable.from_numpy_array(np.ones(shape_stat))
    beta_tmp = nn.Variable.from_numpy_array(np.zeros(shape_stat))
    mean_tmp = nn.Variable.from_numpy_array(np.ones(shape_stat))
    var_tmp = nn.Variable.from_numpy_array(np.ones(shape_stat))
    h = F.batch_normalization(inp, beta_tmp, gamma_tmp, mean_tmp, var_tmp, batch_stat=True)
    return gamma * h + beta


def convolution(x, n, kernel, stride, pad):
    """Convolution wapper"""
    s = nn.initializer.calc_normal_std_glorot(x.shape[1], n, kernel=kernel)
    init = nn.initializer.NormalInitializer(s)
    x = PF.convolution(x, n, kernel=kernel, stride=stride,
                       pad=pad, with_bias=True, w_init=init)
    return x


def deconvolution(x, n, kernel, stride, pad):
    """Convolution wapper"""
    s = nn.initializer.calc_normal_std_glorot(x.shape[1], n, kernel=kernel)
    init = nn.initializer.NormalInitializer(s)
    x = PF.deconvolution(x, n, kernel=kernel, stride=stride,
                         pad=pad, with_bias=True, w_init=init)
    return x


def convblock(x, z=None, n=0, k=(4, 4), s=(2, 2), p=(1, 1), leaky=False):
    """Convolution block"""
    x = convolution(x, n=n, kernel=k, stride=s, pad=p)
    x = CIN(x, z) if z is not None else IN(x, fix_parameters=True)
    x = F.leaky_relu(x, alpha=0.2) if leaky else F.relu(x, inplace=True)
    return x


def unpool_block(x, z=None, n=0, k=(4, 4), s=(2, 2), p=(1, 1), leaky=False, 
                 unpool=False):
    """Deconvolution block"""
    if not unpool:
        logger.info("Deconvolution was used.")
        x = deconvolution(x, n=n, kernel=k, stride=s,
                          pad=p)
    else:
        logger.info("Unpooling was used.")
        x = F.unpooling(x, kernel=(2, 2))
        x = convolution(x, n, kernel=(3, 3), stride=(1, 1),
                        pad=(1, 1))
    x = CIN(x, z) if z is not None else IN(x, fix_parameters=True)
    x = F.leaky_relu(x, alpha=0.2) if leaky else F.relu(x, inplace=True)
    return x


def resblock(x, z=None, n=256):
    """Residual block"""
    r = x
    with nn.parameter_scope('block1'):
        r = convolution(r, n, kernel=(3, 3), pad=(1, 1),
                        stride=(1, 1))
        r = CIN(r, z) if z is not None else IN(r, fix_parameters=True)
        r = F.relu(r, inplace=True)
    with nn.parameter_scope('block2'):
        r = convolution(r, n, kernel=(3, 3), pad=(1, 1),
                        stride=(1, 1))
        r = CIN(r, z) if z is not None else IN(r, fix_parameters=True)

    # Can not use inplace here due to the network architecture
    h = F.relu(x + r, inplace=True)
    return h


"""
Use x and y for avoiding a_real and b_real since these seem like a constnat.
Correspondence to the original paper:

---------------
G_AB  ->  G_XY
G_BA  ->  G_YX
D_A   ->  D_X
D_B   ->  D_Y

E_a   ->  E_x
E_b   ->  E_y
D_Z_A ->  D_z_x
D_Z_B ->  D_z_y
---------------
"""

def generator(x, z, scopename, maps=32, unpool=False):
    """Generator"""
    #TODO: with_bias = True?
    with nn.parameter_scope('generator'):
        with nn.parameter_scope(scopename):
            with nn.parameter_scope('conv1'):
                x = convblock(x, z, n=maps, k=(7, 7), s=(1, 1), p=(3, 3), leaky=False)
            with nn.parameter_scope('conv2'):
                x = convblock(x, z, n=maps*2, k=(3, 3), s=(1, 1), p=(1, 1), leaky=False)
            with nn.parameter_scope('conv3'):
                x = convblock(x, z, n=maps*4, k=(3, 3), s=(2, 2), p=(1, 1), leaky=False)
            for i in range(3):  # Resblocks
                with nn.parameter_scope('res{}'.format(i+1)):
                    x = resblock(x, z, n=maps*4)
            with nn.parameter_scope('deconv1'):
                x = unpool_block(x, z,  n=maps*2, k=(4, 4), s=(2, 2), p=(1, 1),
                                 leaky=False, unpool=unpool)
            with nn.parameter_scope('deconv2'):
                x = convblock(x, z,  n=maps, k=(3, 3), s=(1, 1), p=(1, 1), leaky=False)
            with nn.parameter_scope('last'):
                x = convolution(x, 3, kernel=(7, 7), stride=(1, 1), pad=(3, 3))
            x = F.tanh(x)
    return x


def discriminator(x, scopename, maps=32, kernel=(4, 4)):
    """Discriminator for Sample"""
    with nn.parameter_scope('discriminator'):
        with nn.parameter_scope(scopename):
            with nn.parameter_scope('conv1'):
                x = convolution(x, maps, kernel=kernel, pad=(1, 1), stride=(2, 2))
                x = F.leaky_relu(x, alpha=0.2)
            with nn.parameter_scope('conv2'):
                x = convblock(x, n=maps*2, k=kernel, s=(2, 2), p=(1, 1), leaky=True)
            with nn.parameter_scope('conv3'):
                x = convblock(x, n=maps*4, k=kernel, s=(2, 2), p=(1, 1), leaky=True)
            with nn.parameter_scope('conv4'):
                x = convblock(x, n=maps*4, k=kernel, s=(1, 1), p=(1, 1), leaky=True)
            with nn.parameter_scope('conv5'):
                x = convolution(x, 1, kernel=kernel, pad=(0, 0), stride=(1, 1))
    return x


def G_xy(x, z, unpool=False):
    """Generator from X, Z to Y"""
    return generator(x, z, scopename='xz-y', unpool=unpool)


def G_yx(y, z, unpool=False):
    """Generator from Y, Z to X"""
    return generator(y, z, scopename='yz-x', unpool=unpool)


def D_x(x, maps=32, kernel=(4, 4)):
    return discriminator(x, scopename='x', maps=maps, kernel=kernel)


def D_y(y, maps=32, kernel=(4, 4)):
    return discriminator(y, scopename='y', maps=maps, kernel=kernel)


def encoder(x, y, scopename, latent, maps=32, unpool=False):
    """Encoder
    """
    h = F.concatenate(*[x, y], axis=1)
    with nn.parameter_scope('encoder'):
        with nn.parameter_scope(scopename):
            with nn.parameter_scope('conv1'):
                h = convolution(h, maps, kernel=(3, 3), stride=(2, 2), pad=(1, 1))
                h = F.relu(h, inplace=True)
            with nn.parameter_scope('conv2'):
                h = convblock(h, n=maps*2, k=(3, 3), s=(2, 2), p=(1, 1), leaky=False)
            with nn.parameter_scope('conv3'):
                h = convblock(h, n=maps*4, k=(3, 3), s=(2, 2), p=(1, 1), leaky=False)
            with nn.parameter_scope('conv4'):
                h = convblock(h, n=maps*8, k=(3, 3), s=(2, 2), p=(1, 1), leaky=False)
            with nn.parameter_scope('conv5'):
                h = convblock(h, n=maps*8, k=(4, 4), s=(1, 1), p=(0, 0), leaky=False)
            with nn.parameter_scope('affine'):
                h = PF.affine(h, latent)
    return h


def E(x, y, latent, maps=32, unpool=False):
    """Encoder for x and y"""
    return encoder(x, y, scopename='xy', latent=latent, maps=maps, 
                   unpool=unpool)


def discriminator_latent(z, scopename, maps=32):
    """Discriminator for encoder
    """
    h = z
    with nn.parameter_scope('discriminator-latent'):
        with nn.parameter_scope(scopename):
            with nn.parameter_scope("affine1"):
                h = PF.affine(h, maps)
                h = PF.batch_normalization(h, batch_stat=True)
                h = F.leaky_relu(h, 0.2)
            with nn.parameter_scope("affine2"):
                h = PF.affine(h, maps)
                h = PF.batch_normalization(h, batch_stat=True)
                h = F.leaky_relu(h, 0.2)
            with nn.parameter_scope("affine3"):
                h = PF.affine(h, maps)
                h = PF.batch_normalization(h, batch_stat=True)
                h = F.leaky_relu(h, 0.2)
            h = PF.affine(h, 1, name="last")
    return h

def D_z_x(x, maps=32):
    return discriminator_latent(x, scopename='x', maps=maps)


def image_augmentation(image):
    return F.image_augmentation(image,
                                shape=image.shape,
                                min_scale=1.0,
                                max_scale=286.0/256.0,  # == 1.1171875
                                flip_lr=True,
                                seed=rng_seed)


def recon_loss(x, y):
    return F.mean(F.absolute_error(x, y))


def lsgan_loss(d_fake, d_real=None, persistent=True):
    if d_real:  # Discriminator loss
        loss_d_real = F.mean(F.pow_scalar(d_real - 1., 2.))
        loss_d_fake = F.mean(F.pow_scalar(d_fake, 2.))
        loss = (loss_d_real + loss_d_fake) * 0.5
        loss.persistent = persistent
        return loss
    else:  # Generator loss, this form leads to minimization
        loss = F.mean(F.pow_scalar(d_fake - 1., 2.))
        loss.persistent = persistent
        return loss

def gauss_reparameterize(mu, logvar, eps=1e-2):
    std = F.pow_scalar(F.exp(logvar), 0.5)
    eps = F.randn(0, eps, mu.shape)
    z = mu + eps * std
    #TODO: clip
    return h


def log_prob_laplace(z, mu, logvar):
    #TODO: check why
    sd = F.exp(0.5 * logvar)
    res = - 0.5 * logvar - (F.abs(z - mu) / sd)
    res = res + nn.Variable.from_numpy_array(np.asarray([-np.log(2)]))
    return res


def log_prob_gaussian(z, mu, logvar):
    res = - 0.5 * logvar - (F.pow_scalar(z - mu, 2) / (2.0 * F.exp(logvar)))
    res = res - 0.5 * nn.Variable.from_numpy_array(np.asarray([2 * np.pi]))
    return res


def kld_std_gauss(mu, logvar):
    kld = - 0.5 * F.sum(logvar + 1.0 - F.pow_scalar(mu, 2.0) - F.exp(logvar), axis=(1, ))
    return kld


def main():
    # Check the shape of the output of the generator and the discriminator for images
    print("# Check the shape of the output of the generator and the discriminator for images")
    b, c, h, w = 4, 3, 256, 256
    latent = 16
    x = nn.Variable([b, c, h, w])
    z = nn.Variable([b, 16])
    y = G_xy(x, z)
    print("generator output: ", y.shape)

    y = nn.Variable([b, c, h, w])
    x = G_yx(y, z)
    print("generator output: ", x.shape)

    d_x = D_x(x)
    print("discriminator output: ", d_x.shape)

    
    # Check the shape of the output of the encoder and the discriminator for latent variables
    print("# Check the shape of the output of the encoder and the discriminator for latent variables")
    x = nn.Variable([b, c, h, w])
    y = nn.Variable([b, c, h, w])
    z = E(x, y, latent)
    print("encoder outputs: ", z.shape)  #TODO: shape should be latent

    d_z_z = D_z_x(z)
    print("discriminator output: ", d_z_z.shape)

if __name__ == '__main__':
    main()

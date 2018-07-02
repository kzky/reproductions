import os
import numpy as np
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save
from nnabla.parameter import get_parameter_or_create
from nnabla.ext_utils import get_extension_context
from nnabla.parametric_functions import parametric_function_api

from nnabla.initializer import (
    calc_uniform_lim_glorot,
    ConstantInitializer, NormalInitializer, UniformInitializer)


def spectral_normalization_for_conv(w, itr=1, eps=1e-12):
    d0 = w.shape[0]            # Out
    d1 = np.prod(w.shape[1:])  # In
    w = F.reshape(w, [d0, d1], inplace=False)
    u0 = get_parameter_or_create("singular-vector", [d0], NormalInitializer(), False)
    # Power method
    for _ in range(itr):
        u0 = F.reshape(u0, [1, d0])
        v = F.affine(u0, w)
        v = F.div2(v, F.pow_scalar(F.sum(F.pow_scalar(v, 2.), keepdims=True) + eps, 0.5))
        v = F.reshape(v, [d1, 1])
        u1 = F.affine(w, v)
        u1 = F.div2(u1, F.pow_scalar(F.sum(F.pow_scalar(u1, 2.), keepdims=True) + eps, 0.5))
        u1 = F.reshape(u1, [1, d0])
        u0.data = u1.data  # share buffer
        u1.persistent = False
        u1.need_grad = False
    Wv = F.affine(w, v)
    sigma = F.affine(u1, Wv)
    return sigma

def spectral_normalization_for_affine(w, itr=1, eps=1e-12, input_axis=1):
    d0 = np.prod(w.shape[0:input_axis])  # IN
    d1 = np.prod(w.shape[input_axis:])   # Out
    u0 = get_parameter_or_create("singular-vector", [d1], NormalInitializer(), False)
    # Power method
    for _ in range(itr):
        u0 = F.reshape(u0, [d1, 1])
        v = F.affine(w, u0)
        v = F.div2(v, F.pow_scalar(F.sum(F.pow_scalar(v, 2.), keepdims=True) + eps, 0.5))
        v = F.reshape(v, [1, d0])
        u1 = F.affine(v, w)
        u1 = F.div2(u1, F.pow_scalar(F.sum(F.pow_scalar(u1, 2.), keepdims=True) + eps, 0.5))
        u1 = F.reshape(u1, [d1, 1])
        u0.data = u1.data  # share buffer
        u1.persistent = False
        u1.need_grad = False
    Wv = F.affine(v, w)
    sigma = F.affine(Wv, u1)
    return sigma

def sn_convolution():
    pass

def sn_affine():
    pass

def sn_embed():
    pass


def BN(h, test=False):
    """Batch Normalization"""
    return PF.batch_normalization(h, batch_stat=not test)


def CCBN(h, y, n_classes, test=False):
    """Categorical Conditional Batch Normaliazation"""
    # Call the batch normalization once
    shape_stat = [1 for _ in h.shape]
    shape_stat[1] = h.shape[1]
    gamma_tmp = nn.Variable.from_numpy_array(np.ones(shape_stat))
    beta_tmp = nn.Variable.from_numpy_array(np.zeros(shape_stat))
    mean = get_parameter_or_create(
        "mean", shape_stat, ConstantInitializer(0), False)
    var = get_parameter_or_create(
        "var", shape_stat, ConstantInitializer(0), False)
    h = F.batch_normalization(h, beta_tmp, gamma_tmp, mean, var, batch_stat=not test)

    # Condition the gamma and beta with the class label
    b, c = h.shape[0:2]
    with nn.parameter_scope("gamma"):
        gamma = PF.embed(y, n_classes, c)
        gamma = F.reshape(gamma, [b, c] + [1 for _ in range(len(h.shape[2:]))])
        gamma = F.broadcast(gamma, h.shape)
    with nn.parameter_scope("beta"):
        beta = PF.embed(y, n_classes, c)
        beta = F.reshape(beta, [b, c] + [1 for _ in range(len(h.shape[2:]))])
        beta = F.broadcast(beta, h.shape)
    return gamma * h + beta


def convblock(h, scopename, maps, kernel, pad=(1, 1), stride=(1, 1), upsample=True, test=False):
    with nn.parameter_scope(scopename):
        if upsample:
            h = F.unpooling(h, kernel=(2, 2))
        h = PF.convolution(h, maps, kernel=kernel, pad=pad, stride=stride)
        h = PF.batch_normalization(h, batch_stat=not test)
    return h

@parametric_function_api("attn")
def attnblock(h, r=8, fix_parameters=False):
    """Attention block"""
    x = h

    # 1x1 convolutions
    b, c, s0, s1 = h.shape
    c_r = c // r
    assert c_r > 0
    f_x = PF.convolution(h, c_r, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="f")
    g_x = PF.convolution(h, c_r, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="g")
    h_x = PF.convolution(h, c, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="h")

    # Attend 
    attn = F.batch_matmul(f_x.reshape([b, c_r, -1]), g_x.reshape([b, c_r, -1]), transpose_a=True)
    attn = F.softmax(attn, 1)
    h_x = h_x.reshape([b, c, -1])
    o = F.batch_matmul(h_x, attn)
    o = F.reshape(o, [b, c, s0, s1])

    # Shortcut
    gamma = get_parameter_or_create("gamma", [1, 1, 1, 1], ConstantInitializer(0), not fix_parameters)
    y = gamma * o + x
    return y

def resblock_g(h, y, scopename, 
               n_classes, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), 
               upsample=True, test=False):
    """Residual block for generator"""
    s = h

    with nn.parameter_scope(scopename):
        # BN -> Relu -> Upsample -> Conv
        with nn.parameter_scope("conv1"):
            h = CCBN(h, y, n_classes, test=test)
            h = F.relu(h)
            if upsample:
                h = F.unpooling(h, kernel=(2, 2))
            h = PF.convolution(h, maps, kernel=kernel, pad=pad, stride=stride)
        
        # BN -> Relu -> Conv
        with nn.parameter_scope("conv2"):
            h = CCBN(h, y, n_classes, test=test)
            h = F.relu(h)
            h = PF.convolution(h, maps, kernel=kernel, pad=pad, stride=stride)
            
        # Shortcut: Upsample -> Conv
        with nn.parameter_scope("shortcut"):
            if upsample:
                s = F.unpooling(s, kernel=(2, 2))
            s = PF.convolution(s, maps, kernel=kernel, pad=pad, stride=stride)
    return F.add2(h, s, inplace=True)  #TODO: inplace is permittable?

def resblock_d(h, y, scopename,
               n_classes, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), 
               downsample=True, test=False):
    """Residual block for discriminator"""
    s = h

    with nn.parameter_scope(scopename):
        # BN -> Relu -> Conv
        with nn.parameter_scope("conv1"):
            h = CCBN(h, y, n_classes, test=test)
            h = F.relu(h)
            h = PF.convolution(h, maps, kernel=kernel, pad=pad, stride=stride)
        
        # BN -> Relu -> Conv -> Downsample
        with nn.parameter_scope("conv2"):
            h = CCBN(h, y, n_classes, test=test)
            h = F.relu(h)
            h = PF.convolution(h, maps, kernel=kernel, pad=pad, stride=stride)
            if downsample:
                h = F.average_pooling(h, kernel=(2, 2))
            
        # Shortcut: Conv -> Downsample
        with nn.parameter_scope("shortcut"):
            s = PF.convolution(s, maps, kernel=kernel, pad=pad, stride=stride)
            if downsample:
                s = F.average_pooling(s, kernel=(2, 2))
    return F.add2(h, s, inplace=True)  #TODO: inplace is permittable?


def generator(z, y, scopename="generator", 
              maps=1024, n_classes=1000, s=4, L=5, test=False):
    with nn.parameter_scope(scopename):
        # Affine
        h = PF.affine(z, maps * s * s)
        h = F.reshape(h, [h.shape[0]] + [maps, s, s])

        # Resblocks
        h = resblock_g(h, y, "block-1", n_classes, maps, test=not test)
        h = resblock_g(h, y, "block-2", n_classes, maps // 2, test=not test)
        h = resblock_g(h, y, "block-3", n_classes, maps // 4, test=not test)
        h = resblock_g(h, y, "block-4", n_classes, maps // 8, test=not test)
        h = resblock_g(h, y, "block-5", n_classes, maps // 16, test=not test)

        # Last convoltion
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h)
        h = PF.convolution(h, 3, kernel=(3, 3), pad=(1, 1), stride=(1, 1))
        x = F.tanh(h)
    return x


def discriminator(x, y, scopename="discriminator", 
                  maps=64, n_classes=1000, s=4, L=5, test=False):
    with nn.parameter_scope(scopename):
        # Resblocks
        h = resblock_d(x, y, "block-1", n_classes, maps, test=not test)
        h = resblock_d(h, y, "block-2", n_classes, maps * 2, test=not test)
        h = resblock_d(h, y, "block-3", n_classes, maps * 4, test=not test)
        h = resblock_d(h, y, "block-4", n_classes, maps * 8, test=not test)
        h = resblock_d(h, y, "block-5", n_classes, maps * 16, test=not test)
        h = resblock_d(h, y, "block-6", n_classes, maps * 16, test=not test)

        # Last affine
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h)
        h = F.average_pooling(h, kernel=h.shape[2:])
        o0 = PF.affine(h, 1)

        # Project discriminator
        h = F.reshape(h, h.shape[0:2], inplace=False)
        e = PF.embed(y, n_classes, h.shape[1], name="project-discriminator")
        o1 = F.sum(h * e, axis=1, keepdims=True)
    return o0 + o1


if __name__ == '__main__':
    b, c, h, w = 4, 3, 128, 128
    latent = 128

    print("Generator shape")
    z = F.randn(shape=[b, latent])
    y = nn.Variable([b])
    x = generator(z, y)
    print("x.shape = {}".format(x.shape))

    print("Discriminator shape")
    d = discriminator(x, y)
    print("d.shape = {}".format(d.shape))
    

    print("Attention block")
    b, c, h, w = 4, 32, 128, 128
    x = nn.Variable([b, c, h//2, w//2])
    h = attnblock(x)
    print("h.shape = {}".format(h.shape))
    nn.clear_parameters()
    
    print("Spectral Normalization for Conv")
    o, i, k0, k1 = 8, 8, 16, 16
    w = nn.Variable([o, i, k0, k1])
    sigma = spectral_normalization_for_conv(w, itr=2)
    print("sigma.shape = {}".format(sigma))
    nn.clear_parameters()

    print("Spectral Normalization for Affine")
    o, i = 16, 8
    w = nn.Variable([o, i])
    sigma = spectral_normalization_for_affine(w, itr=2)
    print("sigma.shape = {}".format(sigma))
    nn.clear_parameters()

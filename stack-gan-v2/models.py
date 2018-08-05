import os
import functools
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

import nnabla.initializer as I

#TODO: both conditioned and unconditioned

def to_RGB(x, name="to-RGB"):
    with nn.parameter_scope(name):
        h = PF.convolution(x, 3, kernel=(3, 3), pad=(1, 1))
        h = F.tanh(h)
        return h


def convblock(x, maps, act="relu", test=False, name="convblock"):
    b, c, s0, s1 = x.shape
    with nn.parameter_scope(name):
        h = PF.convolution(x, maps, kernel=(3, 3), pad=(1, 1), with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h, True) if act == "relu" else F.leaky_relu(h, 0.2)
    return h


def upblock(x, maps, test=False, name="upblock"):
    b, c, s0, s1 = x.shape
    with nn.parameter_scope(name):
        h = F.unpooling(x, (2, 2))
        h = PF.convolution(h, maps, kernel=(3, 3), pad=(1, 1), with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h, True)
    return h


def resblock(x, test=False, name="resblock"):
    b, c, s0, s1 = x.shape
    s = x
    with nn.parameter_scope(name):
        h = PF.convolution(x, c * 2, kernel=(3, 3), pad=(1, 1), name="conv-1")  #todo: why 2 times?
        h = PF.batch_normalization(h, batch_stat=not test, name="bn-1")
        h = F.relu(h, True)
        h = PF.convolution(h, c, kernel=(3, 3), pad=(1, 1), name="conv-2")
        h = PF.batch_normalization(h, batch_stat=not test, name="bn-2")
    h = F.add2(h, s, True)
    return h


def conditional_augmentation(e, name="conditional-augmentation"):
    b, c = e.shape
    with nn.parameter_scope(name):
        mu = PF.affine(e, c, name="mu")
        logvar = PF.affine(e, c, name="logvar")
        r = F.randn(0.0, 1.0, shape=(b, c))
        h = mu + r * logvar
    return h


def joint_convblock(x, ce, test=False, name="joint-conv"):
    b, c, s0, s1 = x.shape
    b, d, _, _ = ce.shape
    with nn.parameter_scope(name):
        ce = F.broadcast(ce, [b, d, s0, s1])
        h = F.concatenate(*[x, ce], axis=1)
        h = PF.convolution(h, c, kernel=(3, 3), pad=(1, 1), with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h, True)
    return h


def first_convblock(z, ce, maps=64 * 16, test=False, name="first-convblock"):
    b, d, _, _ = ce.shape
    with nn.parameter_scope(name):
        with nn.parameter_scope("first-affine"):
            ce = F.broadcast(ce, [b, d, 4, 4])
            h = F.concatenate(*[z, ce], axis=1)
            h = PF.affine(h, (maps * 2) * 4 * 4)
            h = PF.batch_normalization(h, batch_stat=not test)
            h = F.relu(h)
            b, _ = h.shape
            h = F.reshape(h, (b, maps * 2, 4, 4))
        h = upblock(h, maps // 2, test, "upblock-1")
        h = upblock(h, maps // 4, test, "upblock-2")
        h = upblock(h, maps // 8, test, "upblock-3")
        h = upblock(h, maps // 16, test, "upblock-4")
    return h


def stage_convblock(x, ce, test=False, name="stage-x"):
    b, c, s0, s1 = x.shape
    b, d, _, _ = ce.shape
    with nn.parameter_scope(name):
        # Joint-conv
        h = joint_convblock(x, ce, name="joint-convblock")
        # Resblock
        h = resblock(h, test=test, name="resblock-1")
        h = resblock(h, test=test, name="resblock-2")
        # Upsample
        h = upblock(h, c, test)
    return h


def generator(z, ce, maps=64 * 16, test=False, name="generator"):
    # Multi-stage generation
    h = first_convblock(z, ce, maps=maps, test=test)
    img0 = to_RGB(h, name="to-RGB-0")  # 64x64
    h = stage_convblock(h, ce, test, name="stage-1")
    img1 = to_RGB(h, name="to-RGB-1")  # 128x128
    h = stage_convblock(h, ce, test, name="stage-2")
    img2 = to_RGB(h, name="to-RGB-2")  # 256x256
    return [img0, img1, img2]  
    

def downblock(x, maps, bn=True, test=False, name="downblock"):
    with nn.parameter_scope(name):
        h = PF.convolution(x, maps, kernel=(4, 4), stride=(2, 2), pad=(1, 1))
        h = PF.batch_normalization(h, batch_stat=not test) if bn else h
        h = F.leaky_relu(h, 0.2)
    return h


def downblocks(x, maps, itr=3, test=False):
    h = downblock(x, maps, bn=False, test=test, name="downblock-1")
    for i in range(itr):
        c = maps * 2 ** (i + 1)
        h = downblock(x, c, test=test, name="downblock-{}".format(i + 2))
    return h


def discriminator(x, ce, maps=64, itr=3, test=False, name="discrimiator-x"):
    with nn.parameter_scope(name):
        # down sampling
        h = downblocks(x, maps, itr=itr)
        # convs
        if itr > 3:
            c = h.shape[1]
            for i in range(itr - 3):
                c = c // 2
                h = convblock(h, c , act="leaky_relu", test=test, name="conv-{}".format(i))

        # joining
        h = joint_convblock(h, ce, test=test, name="joint-convblock")
        # last conv
        h = PF.convolution(h, 1, kernel=(4, 4), stride=(4, 4), name="last-conv")
    return h


def gan_loss(d_fake, d_real=None):
    if d_real is not None:
        return F.mean(F.log(1.0 - F.sigmoid(d_fake)))
    return - F.mean(F.log(F.sigmoid(d_real))) - F.mean(F.log(1.0 - F.sigmoid(d_fake)))


if __name__ == '__main__':
    # Input
    b, c = 32, 128
    n_words = 500
    z = F.randn(0, 1, shape=[b, c])
    w = nn.Variable.from_numpy_array(np.random.choice(np.arange(n_words), b, replace=True))
    e = PF.embed(w, n_words, c)
    maps = 64
    factor = 16
    test = False
    
    # Conditional embedding
    ce = conditional_augmentation(e)
    ce = F.identity(F.reshape(ce, (b, c, 1, 1)))
    
    # Generator
    imgs = generator(z, ce, maps * factor, test)
    for img in imgs:
        print(img)

    # Discriminators
    d_fakes = []
    for i, img in enumerate(imgs):
        d_fake = discriminator(img, ce, itr=i + 3, name="discriminator-{}".format(64 * 2 ** i))
        print(d_fake)
        d_fakes.append(d_fake)
    
    # for n, v in nn.get_parameters().items():
    #     print(n, v.shape)

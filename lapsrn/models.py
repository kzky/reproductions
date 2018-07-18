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

from nnabla.initializer import (
    calc_uniform_lim_glorot,
    ConstantInitializer, NormalInitializer, UniformInitializer)


def rblock(x, name, maps=64, kernel=(3, 3), pad=(1, 1), stride=(1, 1), r=0, D=5, bn=False, test=False):
    h = x
    for d in range(D):
        with nn.parameter_scope("recursive-block-{}-{}".format(d, name)):
            # Relu -> Conv -> (BN)
            h = F.relu(h)
            h = PF.convolution(h, maps, kernel, pad, stride)
            h = h if not bn else PF.batch_normalization(h, batch_stat=not test, name="bn-{}".format(r))
    return h + x


def upsample(x, name, maps=64, kernel=(4, 4), pad=(1, 1), stride=(2, 2)):
    return PF.deconvolution(x, maps, kernel, pad, stride, name=name)


def feature_extractor(x, name, maps=64, kernel=(3, 3), pad=(1, 1), stride=(1, 1), R=8, D=5,
                      bn=False, test=False):
    h = x
    with nn.parameter_scope("feature-extractor-{}".format(name)):
        # {Relu -> Conv -> (BN)} x R -> upsample -> conv
        for r in range(R):
            h = rblock(h, "shared", maps, kernel, pad, stride, r=r, D=D, bn=bn, test=test)
        u = upsample(h, "shared", maps)
        r = PF.convolution(u, 3, name="shared", kernel=kernel, pad=pad, stride=stride)
    return u, r

def lapsrn(x_l, maps=64, S=3, R=8, D=5, bn=False, test=False):
    u_irb = x_l
    u_feb = PF.convolution(x_l, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="first-conv")
    for s in range(S):
        u_feb, r = feature_extractor(u_feb, "shared", maps, R=R, D=D, bn=bn, test=test)
        u_irb = upsample(u_irb, "shared", 3) + r
    x_h = PF.convolution(u_irb, 3, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="last-conv")
    return x_h

def loss_charbonnier(x, y, eps=1e-3):
    lossv = F.pow_scalar((y - x) ** 2 + eps ** 2, )
    return lossv


def get_loss(loss):
    if loss == "charbonnier":
        loss_func = loss_charbonnier
    elif loss == "l2":
        loss_func = F.squared_loss
    elif loss == "l1":
        loss_func = F.absolute_loss
    else:
        raise ValueError("Loss is not supported.")
    return loss

if __name__ == '__main__':
    x_l = nn.Variable([4, 3, 16, 16])
    x_h = lapsrn(x_l, 64)
    print(x_h)

    for n, v in nn.get_parameters().items():
        print(n, v.shape)

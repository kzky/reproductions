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


def rblock(x, maps=64, kernel=(3, 3), pad=(1, 1), stride=(1, 1), 
           r=0, D=5, bn=False, test=False, name=None):
    h = x
    for d in range(D):
        with nn.parameter_scope("recursive-block-{}-{}".format(d, name)):
            # LeakyRelu -> Conv -> (BN)
            h = F.leaky_relu(h, 0.2)
            h = PF.convolution(h, maps, kernel, pad, stride)
            h = h if not bn \
                else PF.batch_normalization(h, batch_stat=not test, name="bn-{}".format(r))
    return h + x


def upsample(x, maps=64, kernel=(4, 4), pad=(1, 1), stride=(2, 2), name=None):
    with nn.parameter_scope("upsample-{}".format(name)):
        return PF.deconvolution(x, maps, kernel, pad, stride)


def residue(x, maps=3, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name=None):
    with nn.parameter_scope("residue-{}".format(name)):
        return PF.convolution(x, maps, kernel, pad, stride)


def feature_extractor(x, maps=64, kernel=(3, 3), pad=(1, 1), stride=(1, 1), 
                      R=8, D=5, skip_type="ss", bn=False, test=False, name=None):
    h = x
    s = h
    with nn.parameter_scope("feature-extractor-{}".format(name)):
        # {Relu -> Conv -> (BN)} x R -> upsample -> conv
        for r in range(R):
            h = rblock(h, maps, kernel, pad, stride, r=r, D=D, bn=bn, test=test, name="shared")
            if skip_type == "ss":
                h = h + x
            elif skip_type == "ds":
                h = h + s
                s = h
            elif skip_type == "ns":
                h = h
        u = upsample(h, maps, name="shared")
        r = residue(u, 3, kernel, pad, stride, name="shared")
    return u, r

def lapsrn(x, maps=64, S=3, R=8, D=5, skip_type="ss", bn=False, test=False):
    u_irbs = []
    u_irb = x
    u_feb = PF.convolution(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="first-conv")
    for s in range(S):
        u_feb, r = feature_extractor(u_feb, maps, R=R, D=D, 
                                     skip_type=skip_type, bn=bn, test=test, name="shared")
        u_irb = upsample(u_irb, 3, name="shared") + r
        u_irbs.append(u_irb)
    return u_irbs

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

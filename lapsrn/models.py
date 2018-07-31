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

class BilinearUpsampleInitiazlier(I.BaseInitializer):

    def __init__(self, imap=64, omap=1, kernel=(4, 4)):
        assert kernel[0] == kernel[1]

        k = kernel[0]
        f = (k + 1) // 2
        if k % 2 == 1:
            c = f - 1
        else:
            c = f - 0.5
        og = np.ogrid[:k, :k]
        k = (1 - abs(og[0] - c) // f) * (1 - abs(og[1] - c) // f)
        
        self.w_init = np.zeros((omap, imap) + kernel).astype(np.float32)
        self.w_init[:omap, :imap, :, :] = k


    def __call__(self, ):
        return self.w_init

BUI = BilinearUpsampleInitiazlier

def convolution(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name=None):
    std = I.calc_normal_std_he_forward(x.shape[1], maps, kernel)
    initizlier = I.NormalInitializer(std)
    return PF.convolution(x, maps, kernel, pad, stride, name=name, with_bias=False)


def block(x, maps=64, kernel=(3, 3), pad=(1, 1), stride=(1, 1), 
           r=0, D=5, bn=False, test=False, name=None):
    h = x
    for d in range(D):
        if name == "across-pyramid":
            scopename = "block-{}-{}".format(r, d)
        else:  # within-pyramid
            scopename = "block-{}".format(d)
        with nn.parameter_scope(scopename):
            # LeakyRelu -> Conv -> (BN)
            h = F.leaky_relu(h, 0.2)
            h = convolution(h, maps, kernel, pad, stride)
            h = h if not bn \
                else PF.batch_normalization(h, batch_stat=not test, name="bn-{}".format(r))
    return h + x


def upsample(x, maps=64, kernel=(4, 4), pad=(1, 1), stride=(2, 2), 
             initializer=None, name=None):
    if not initializer:
        std = I.calc_normal_std_he_forward(x.shape[1], maps, kernel)
        initizlier = I.NormalInitializer(std)
    with nn.parameter_scope("upsample-{}".format(name)):
        return PF.deconvolution(x, maps, kernel, pad, stride, with_bias=False)


def residue(x, maps=3, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name=None):
    with nn.parameter_scope("residue-{}".format(name)):
        return convolution(x, maps, kernel, pad, stride)


def feature_extractor(x, maps=64, kernel=(3, 3), pad=(1, 1), stride=(1, 1), 
                      R=8, D=5, skip_type="ss", bn=False, test=False, name=None):
    h = x
    s = h
    with nn.parameter_scope("feature-extractor-{}".format(name)):
        # {Relu -> Conv -> (BN)} x R -> upsample -> conv
        for r in range(R):
            h = block(h, maps, kernel, pad, stride, r=r, D=D, bn=bn, test=test, 
                      name=name)
            if skip_type == "ss":
                h = h + x
            elif skip_type == "ds":
                h = h + s
                s = h
            elif skip_type == "ns":
                h = h
        u = upsample(h, maps, name=name, initializer=BUI(h.shape[1], 1))
        r = residue(u, 1, kernel, pad, stride, name=name)
    return u, r


def lapsrn(x, maps=64, S=3, R=8, D=5, skip_type="ss", 
           bn=False, test=False, share_type="across-pyramid"):
    u_irbs = []
    u_irb = x
    u_feb = convolution(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="first-conv")
    for s in range(S):
        name = share_type if share_type == "across-pyramid" else str(s)
        u_feb, r = feature_extractor(u_feb, maps, R=R, D=D, 
                                     skip_type=skip_type, bn=bn, test=test, 
                                     name=name)
        u_irb = upsample(u_irb, 1, initializer=BUI(u_feb.shape[1], 1), name=name) + r
        u_irbs.append(u_irb)
    return u_irbs


def loss_charbonnier(x, y, eps=1e-3):
    lossv = F.pow_scalar(F.squared_error(x, y) + eps ** 2, 0.5)
    return lossv


def get_loss(loss):
    if loss == "charbonnier":
        loss_func = loss_charbonnier
    elif loss == "l2":
        loss_func = F.squared_error
    elif loss == "l1":
        loss_func = F.absolute_error
    else:
        raise ValueError("Loss ({}) is not supported.".format(loss))
    return loss_func

if __name__ == '__main__':
    # Across pyramid
    x_l = nn.Variable([4, 3, 16, 16])
    x_h = lapsrn(x_l, 64, share_type="across-pyramid")
    print(x_h)

    for n, v in nn.get_parameters().items():
        print(n, v.shape)
    nn.clear_parameters()
    

    # Within pyramid
    x_l = nn.Variable([4, 3, 16, 16])
    x_h = lapsrn(x_l, 64, share_type="within-pyramid")
    print(x_h)

    for n, v in nn.get_parameters().items():
        print(n, v.shape)

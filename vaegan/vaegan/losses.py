import numpy as np
import chainer
import chainer.variable as variable
from chainer.functions.activation import lstm
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer.cuda import cupy as cp
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from collections import OrderedDict
import logging
import time
from vaegan import mlp_models

class ReconstructionLoss(Chain):
    def __init__(self, ):
        pass

    def __call__(self, x, y):
        d = np.prod(x.shape[1:])
        l = F.mean_squared_error(x, y) / d
        return l

class GANLoss(Chain):

    def __init__(self, ):
        pass

    def __call__(self, d_x, d_x_rec, d_x_gen):
        bs_d_x = d_x.shape[0]
        bs_d_x_rec = d_x_rec.shape[0]
        bs_d_x_gen = d_x_gen.shape[0]

        l = F.sum(F.log(d_x)) / bs_d_x \
            + F.sum(F.log(1 - d_x_rec)) / bs_d_x_rec \
            + F.sum(F.log(1 - d_x_gen)) / bs_d_x_gen
        return l

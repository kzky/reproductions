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

def generate_norm(bs, dim=100, sigma=0.43, device=None):
    """Generate samples following normal distribution, ranging from -1 to +1
    """
    r_ = (np.random.randn(bs, dim) * sigma).astype(np.float32)
    if device:
        r = Variable(cuda.to_gpu(r_, device))
    else:
        r = Variable(r_)
    return r

def generate_unif(bs, dim=100, device=None):
    """Generate samples following uniform distribution, ranging from -1 to +1
    """
    r_ = (np.random.rand(bs, dim) *2. - 1.).astype(np.float32)
    if device:
        r = Variable(cuda.to_gpu(r_, device))
    else:
        r = Variable(r_)
    return r
    
def calc_conv(n, p=1, k=4, s=2):
    return (n + 2 * p - k) / s + 1

def calc_deconv(n, p=1, k=4, s=2):
    return s * (n - 1) + k - 2 * p

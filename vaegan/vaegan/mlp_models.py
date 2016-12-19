"""
VAEGAN w/o VAE-loss
"""
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

class EncNet(Chain):
    def __init__(self, dim, top=False, act=F.relu, device=None):
        d_inp, d_out = dim
        if top:
            self._init_top(d_inp, d_out)
        else:
            self._init_non_top(d_inp, d_out)
        self.top = top
        self.act = act
        self.device = device

    def _init_top(self, d_inp, d_out):
        super(EncNet, self).__init__(
            linear=L.Linear(d_inp, d_out),
            bn=L.BatchNormalization(d_out, decay=0.9,
                                    use_gamma=False, use_beta=False),
        )
        
    def _init_non_top(self, d_inp, d_out):
        super(EncNet, self).__init__(
            linear=L.Linear(d_inp, d_out),
            bn=L.BatchNormalization(d_out, decay=0.9,
                                    use_gamma=False, use_beta=False),
            sb=L.Scale(W_shape=d_out, bias_term=True)
        )

    def __call__(self, h, test=False):
        if self.top:
            h = self.linear(h)
            h = self.bn(h, test)
            h = F.normalize(h)  # Normalize such that h/|h|_2
        else:
            h = self.linear(h)
            h = self.bn(h, test)
            h = self.sb(h)
            h = self.act(h)
        return h

class Encoder(Chain):
    def __init__(self, act=F.relu, device=None):
        super(Encoder, self).__init__(
            en0=EncNet((784, 500), top=False, act=act, device=device),
            en1=EncNet((500, 250), top=False, act=act, device=device),
            en2=EncNet((250, 100), top=True, act=act, device=device),
        )
    
    def __call__(self, x, test=False):
        h = self.en0(x, test)
        h = self.en1(h, test)
        h = self.en2(h, test)
        return h

class DecNet(Chain):
    def __init__(self, dim, bottom=False, act=F.relu, device=None):
        d_inp, d_out = dim
        if bottom:
            self._init_bottom(d_inp, d_out)
        else:
            self._init_non_bottom(d_inp, d_out)
        self.bottom = bottom
        self.act = act
        self.device = device

    def _init_bottom(self, d_inp, d_out):
        super(DecNet, self).__init__(
            linear=L.Linear(d_inp, d_out),
        )

    def _init_non_bottom(self, d_inp, d_out):
        super(DecNet, self).__init__(
            linear=L.Linear(d_inp, d_out),
            bn=L.BatchNormalization(d_out, decay=0.9,
                                    use_gamma=False, use_beta=False),
            sb=L.Scale(W_shape=d_out, bias_term=True)
        )

    def __call__(self, h, test=False):
        if not self.bottom:
            h = self.linear(h)
            h = self.bn(h, test)
            h = self.sb(h)
            h = self.act(h)
        else:
            h = self.linear(h)
            h = F.tanh(h)
        return h

class Decoder(Chain):
    def __init__(self, act=F.relu, device=None):
        super(Decoder, self).__init__(
            dn0=DecNet((100, 250), bottom=False, act=act, device=device),
            dn1=DecNet((250, 500), bottom=False, act=act, device=device),
            dn2=DecNet((500, 784), bottom=True, act=act, device=device),
        )
        self.act = act
        self.device = device

    def __call__(self, z, test=False):
        h = self.dn0(z, test)
        h = self.dn1(h, test)
        h = self.dn2(h, test)
        return h

# Alias
Generator = Decoder

class DisNet(Chain):
    def __init__(self, dim, last=False, act=F.relu, device=None):
        d_inp, d_out = dim
        if last:
            self._init_last(d_inp, d_out)
        else:
            self._init_non_last(d_inp, d_out)
        self.last = last
        self.act = act
        self.device = device

    def _init_last(self, d_inp, d_out):
        super(DisNet, self).__init__(
            linear=L.Linear(d_inp, d_out),
        )
        
    def _init_non_last(self, d_inp, d_out):
        super(DisNet, self).__init__(
            linear=L.Linear(d_inp, d_out),
            bn=L.BatchNormalization(d_out, decay=0.9,
                                    use_gamma=False, use_beta=False),
            sb=L.Scale(W_shape=d_out, bias_term=True)
        )

    def __call__(self, h, test=False):
        if self.last:
            h = self.linear(h)
            h = F.sigmoid(h) 
        else:
            h = self.linear(h)
            h = self.bn(h, test)
            h = self.sb(h)
            h = self.act(h)
        return h

class Discriminator(Chain):
    def __init__(self, act=F.relu, device=None):
        super(Discriminator, self).__init__(
            dn0=DisNet((784, 500), last=False, act=act, device=device),
            dn1=DisNet((500, 250), last=False, act=act, device=device),
            dn2=DisNet((250, 100), last=False, act=act, device=device),
            dn3=DisNet((100, 1), last=True, act=act, device=device), 
        )
        self.hiddens = []

    def __call__(self, x, test=False):
        self.hiddens = []
        
        h = self.dn0(x, test)
        self.hiddens.append(h)
        
        h = self.dn1(h, test)
        self.hiddens.append(h)

        h = self.dn2(h, test)
        self.hiddens.append(h)
        
        h = self.dn3(h, test)
        return h

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

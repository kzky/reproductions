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
from vaegan.chainer_fix import BatchNormalization

class EncNet(Chain):
    def __init__(self, dim, ksize=(4, 4), stride=(2, 2), pad=(1, 1),
                 top=False, act=F.relu, device=None):
        self.top = top
        self.act = act
        self.device = device
        
        d_inp, d_out = dim
        if top:
            self._init_top(d_inp, d_out)
        else:
            self._init_non_top(d_inp, d_out, ksize, stride, pad)

    def _init_top(self, d_inp, d_out):
        super(EncNet, self).__init__(
            linear=L.Linear(d_inp, d_out),
            bn=BatchNormalization(d_out, decay=0.9,
                                  use_gamma=False, use_beta=False,
                                  device=self.device),
        )
        
    def _init_non_top(self, d_inp, d_out, ksize, stride, pad):
        super(EncNet, self).__init__(
            conv=L.Convolution2D(d_inp, d_out, ksize, stride, pad),
            bn=BatchNormalization(d_out, decay=0.9,
                                  use_gamma=False, use_beta=False, 
                                  device=self.device),
            sb=L.Scale(W_shape=d_out, bias_term=True)
        )

    def __call__(self, h, test=False):
        if self.top:
            h = self.linear(h)
            h = self.bn(h, test)
            h = F.normalize(h)  #TODO: Normalize such that h/|h|_2
        else:
            h = self.conv(h)
            h = self.bn(h, test)
            h = self.sb(h)
            h = self.act(h)
        return h

class Encoder(Chain):
    def __init__(self, act=F.relu, device=None):
        super(Encoder, self).__init__(
            en0=EncNet((1, 32), ksize=(4, 4), stride=(2, 2), pad=(1, 1),
                       top=False, act=act, device=device),
            en1=EncNet((32, 64), ksize=(4, 4), stride=(2, 2), pad=(1, 1),
                       top=False, act=act, device=device),
            en2=EncNet((7*7*64, 100), top=True, act=act, device=device),
        )
    
    def __call__(self, x, test=False):
        print("Encoder")
        h = self.en0(x, test)
        print(h.shape)
        h = self.en1(h, test)
        print(h.shape)
        h = self.en2(h, test)
        print(h.shape)
        return h

class DecNet(Chain):
    def __init__(self, dim, ksize=(4, 4), stride=(2, 2), pad=(1, 1),
                 where="middle", act=F.relu, device=None):
        self.where = where
        self.act = act
        self.device = device

        d_inp, d_out = dim
        if where == "top":
            self._init_top(d_inp, d_out)
        elif where == "middle":
            self._init_middle(d_inp, d_out, ksize, stride, pad)
        elif where == "bottom":
            self._init_bottom(d_inp, d_out, ksize, stride, pad)

    def _init_top(self, d_inp, d_out):
        super(DecNet, self).__init__(
            linear=L.Linear(d_inp, d_out),
            bn=BatchNormalization(d_out, decay=0.9,
                                  use_gamma=False, use_beta=False, 
                                  device=self.device),
            sb=L.Scale(W_shape=d_out, bias_term=True)
        )

    def _init_middle(self, d_inp, d_out, ksize, stride, pad):
        super(DecNet, self).__init__(
            deconv=L.Deconvolution2D(d_inp, d_out, ksize, stride, pad),
            bn=BatchNormalization(d_out, decay=0.9,
                                  use_gamma=False, use_beta=False, 
                                  device=self.device),
            sb=L.Scale(W_shape=d_out, bias_term=True)
        )
            
    def _init_bottom(self, d_inp, d_out, ksize, stride, pad):
        super(DecNet, self).__init__(
            deconv=L.Deconvolution2D(d_inp, d_out, ksize, stride, pad),
        )

    def __call__(self, h, test=False):
        if self.where == "top":
            h = self.linear(h)
            h = self.bn(h, test)
            h = self.sb(h)
            h = self.act(h)
            bs = h.shape[0]
            h = F.reshape(h, (bs, 64, 7, 7))
        elif self.where == "middle":
            h = self.deconv(h)
            h = self.bn(h, test)
            h = self.sb(h)
            h = self.act(h)
        elif self.where == "bottom":
            h = self.deconv(h)
            h = F.tanh(h)
        return h

class Decoder(Chain):
    def __init__(self, act=F.relu, device=None):
        self.act = act
        self.device = device
        
        super(Decoder, self).__init__(
            dn0=DecNet((100, 7*7*64), where="top", act=act, device=device),
            dn1=DecNet((64, 32), ksize=(4, 4), stride=(2, 2), pad=(1, 1),
                       where="middle", act=act, device=device),
            dn2=DecNet((32, 1), ksize=(4, 4), stride=(2, 2), pad=(1, 1),
                       where="bottom", act=act, device=device),
        )

    def __call__(self, z, test=False):
        print("Decoder")
        h = self.dn0(z, test)
        print(h.shape)
        h = self.dn1(h, test)
        print(h.shape)
        h = self.dn2(h, test)
        print(h.shape)
        return h

# Alias
Generator = Decoder

class DisNet(Chain):
    def __init__(self, dim, ksize=(4, 4), stride=(2, 2), pad=(1, 1),
                 top=False, act=F.relu, device=None):
        self.top = top
        self.act = act
        self.device = device
        
        d_inp, d_out = dim
        if top:
            self._init_top(d_inp, d_out)
        else:
            self._init_non_top(d_inp, d_out, ksize, stride, pad)

    def _init_top(self, d_inp, d_out):
        super(DisNet, self).__init__(
            linear=L.Linear(d_inp, d_out),
        )
        
    def _init_non_top(self, d_inp, d_out, ksize, stride, pad):
        super(DisNet, self).__init__(
            conv=L.Convolution2D(d_inp, d_out, ksize, stride, pad),
            bn=BatchNormalization(d_out, decay=0.9,
                                  use_gamma=False, use_beta=False, 
                                  device=self.device),
            sb=L.Scale(W_shape=d_out, bias_term=True)
        )

    def __call__(self, h, test=False):
        if self.top:
            h = self.linear(h)
            h = F.sigmoid(h)
        else:
            h = self.conv(h)
            h = self.bn(h, test)
            h = self.sb(h)
            h = self.act(h)
        return h

class Discriminator(Chain):
    def __init__(self, act=F.relu, device=None):
        super(Discriminator, self).__init__(
            dn0=DisNet((1, 32), ksize=(4, 4), stride=(2, 2), pad=(1, 1),
                       top=False, act=act, device=device),
            dn1=DisNet((32, 64), ksize=(4, 4), stride=(2, 2), pad=(1, 1),
                       top=False, act=act, device=device),
            dn2=DisNet((7*7*64, 1), top=True, act=act, device=device),
        )
        self.hiddens = []

    def __call__(self, x, test=False):
        print("Discriminator")
        self.hiddens = []
        
        h = self.dn0(x, test)
        self.hiddens.append(h)
        print(h.shape)
        
        h = self.dn1(h, test)
        self.hiddens.append(h)
        print(h.shape)

        h = self.dn2(h, test)
        self.hiddens.append(h)
        print(h.shape)
        
        return h

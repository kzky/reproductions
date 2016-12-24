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
            en2=EncNet((64, 128), ksize=(4, 4), stride=(2, 2), pad=(1, 1),
                       top=False, act=act, device=device),
            en3=EncNet((3*3*128, 100), top=True, act=act, device=device),
        )
    
    def __call__(self, x, test=False):
        h = self.en0(x, test)
        h = self.en1(h, test)
        h = self.en2(h, test)
        h = self.en3(h, test)
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
            self._init_middle(d_inp, d_out)
        elif where == "bottom":
            self._init_bottom(d_inp, d_out)

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
            dn0=DecNet((100, 3*3*128), where="top", act=act, device=device),
            dn1=DecNet((128, 64), ksize=(4, 4), stride=(2, 2), pad=(1, 1),
                       where="middle", act=act, device=device),
            dn2=DecNet((64, 32), ksize=(4, 4), stride=(2, 2), pad=(1, 1),
                       where="middle", act=act, device=device),
            dn3=DecNet((32, 1), ksize=(4, 4), stride=(2, 2), pad=(1, 1),
                       where="bottom", act=act, device=device),
        )

    def __call__(self, z, test=False):
        h = self.dn0(z, test)
        h = self.dn1(h, test)
        h = self.dn2(h, test)
        h = self.dn3(h, test)
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
            dis0=DisNet((1, 32), ksize=(4, 4), stride=(2, 2), pad=(1, 1),
                       top=False, act=act, device=device),
            dis1=DisNet((32, 64), ksize=(4, 4), stride=(2, 2), pad=(1, 1),
                       top=False, act=act, device=device),
            dis2=DisNet((64, 128), ksize=(4, 4), stride=(2, 2), pad=(1, 1),
                       top=False, act=act, device=device),
            dis3=DisNet((3*3*128, 1), top=True, act=act, device=device),
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

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
from mlp_models import Encoder, Decoder, Discriminator, ReconstructionLoss, GANLoss

class MLPExperiment():
    def __init__(self, act, device=None, learning_rate=0.9, gamma=1.):
        self.device = device
        self.gamma = gamma

        # Model
        self.encoder = Encoder(act=act, device=device)
        self.decoder = Decoder(act=act, device=device)
        self.generator = self.decoder
        self.discriminator = Discriminator(act=act, device=device)

        # To GPU
        if device is not None:
            self.encoder.to_gpu(device)
            self.decoder.to_gpu(device)
            self.discriminator.to_gpu(device)

        # Optimizer
        self.optimizer_enc = optimizers.Adam(learning_rate)
        self.optimizer_dec = optimizers.Adam(learning_rate)
        self.optimizer_gen = self.optimizer_dec
        self.optimizer_dis = optimizers.Adam(learning_rate)
        self.optimizer_enc.use_cleargrads()
        self.optimizer_dec.use_cleargrads()
        self.optimizer_dis.use_cleargrads()
            
    def train(self, x, z_p):
        # Enc/Dec
        z = self.encoder(x)
        x_rec = self.decoder(z)

        # Dis for x and x_rec
        d_x = self.discriminator(x)
        hiddens_x = d_x.hiddens
        d_x_rec = self.discriminator(x_rec)
        hiddens_x_rec = d_x_rec.hiddens

        # Reconstruction for Dis
        l_dis = 0
        for h_d_x, h_d_x_rec in zip(hiddens_x, hiddens_x_rec):
            l_dis += ReconstructionLoss(h_d_x, h_d_x_rec)
        
        # Generate
        bs = z.shape[0]
        z_p = self._generate(bs)
        x_gen = self.generator(z_p)

        # GAN loss
        l_gan = GANLoss(d_x, d_x_rec, d_x_gen)

        # Backward for enc, dec, and dis
        self.cleargrads()
        l_dis.backward()
        self.optimizer_enc.update()

        self.cleargrads()
        l_dec = l_dis + l_gan
        l_dec.backward()
        self.optimizer_dec.update()

        self.cleargrads()
        l_dis = -l_dis  # gradient ascend for discriminator
        l_dis.backward()
        self.optimizer_dis.update()
        
    def test(self, bs=10, x):
        z_p = self._generate(bs)
        x_gen = self.generator(z_p, test=False)

        z = self.encoder(x, test=False)
        x_rec = self.decoder(x, test=False)

        return x_rec, x_gen

    def cleargrads(self, ):
        self.encoder.cleargrads()
        self.decoder.cleargrads()
        self.discriminator.cleargrads()

    def _generate(self, bs=64):
        if self.device:
            z = cp.random.uniform(-1, 1, (bs, 100)).astype(np.float32)
            z = cp.to_gpu(z)
            z = Variable(z, device))
        else:
            z = Variable(np.random.uniform(-1, 1, (bs, 100)).astype(np.float32))
        return z

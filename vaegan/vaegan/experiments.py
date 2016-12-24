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
from vaegan import mlp_models, cnn_models
from vaegan.losses import ReconstructionLoss, GANLoss

"""
Reference
---------------
[1] Anders Boesen Lindbo Larsen, Soren Kaae Sonderby, Hugo Larochelle, Ole Winther: Autoencoding beyond pixels using a learned similarity metric. ICML 2016: 1558-1566
"""

class MLPExperiment():
    def __init__(self, act, device=None, learning_rate=0.9, gamma=1.):
        self.device = device
        self.gamma = gamma

        # Model
        self.encoder = mlp_models.Encoder(act=act, device=device)
        self.decoder = mlp_models.Decoder(act=act, device=device)
        self.generator = self.decoder
        self.discriminator = mlp_models.Discriminator(act=act, device=device)

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
        self.optimizer_enc.setup(self.encoder)
        self.optimizer_dec.setup(self.decoder)
        self.optimizer_dis.setup(self.discriminator)
        self.optimizer_enc.use_cleargrads()
        self.optimizer_dec.use_cleargrads()
        self.optimizer_dis.use_cleargrads()

        # Loss
        self.reconstruction_loss = ReconstructionLoss()
        self.gan_loss = GANLoss()
            
    def train(self, x, z_p):
        # Enc/Dec
        z = self.encoder(x)
        x_rec = self.decoder(z)

        # Dis for x and x_rec
        d_x = self.discriminator(x)
        hiddens_d_x = self.discriminator.hiddens
        d_x_rec = self.discriminator(x_rec)
        hiddens_d_x_rec = self.discriminator.hiddens

        # Reconstruction Loss for Dis
        l_dis = 0
        for h_d_x, h_d_x_rec in zip(hiddens_d_x, hiddens_d_x_rec):
            l_dis += self.reconstruction_loss(h_d_x, h_d_x_rec)
        
        # Generate
        bs = z.shape[0]
        z_p = self._generate(bs)
        x_gen = self.generator(z_p)
        d_x_gen = self.discriminator(x_gen)

        # GAN loss
        l_gan = self.gan_loss(d_x, d_x_rec, d_x_gen)

        # Backward for enc, dec(=gen), and dis
        ## With these update rule,
        ## gradients of GAN Loss do not influence to Enc's parameters update
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
        
    def test(self, x, bs=10):
        """Generate images from data and random seeds
        """
        z = self.encoder(x, test=True)
        x_rec = self.decoder(z, test=True)
        
        z_p = self._generate(bs)
        x_gen = self.generator(z_p, test=True)

        d_x_rec = self.discriminator(x_rec)
        d_x_gen = self.discriminator(x_gen)

        return x_rec, x_gen, d_x_rec, d_x_gen

    def cleargrads(self, ):
        self.encoder.cleargrads()
        self.decoder.cleargrads()
        self.discriminator.cleargrads()

    def _generate(self, bs=64):
        if self.device:
            z = cp.random.uniform(-1, 1, (bs, 100)).astype(np.float32)
            z = cuda.to_gpu(z, self.device)
            z = Variable(z)
        else:
            z = Variable(np.random.uniform(-1, 1, (bs, 100)).astype(np.float32))
        return z

        
class CNNExperiment():
    def __init__(self, act, device=None, learning_rate=0.9, gamma=1.):
        self.device = device
        self.gamma = gamma

        # Model
        self.encoder = cnn_models.Encoder(act=act, device=device)
        self.decoder = cnn_models.Decoder(act=act, device=device)
        self.generator = self.decoder
        self.discriminator = cnn_models.Discriminator(act=act, device=device)

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
        self.optimizer_enc.setup(self.encoder)
        self.optimizer_dec.setup(self.decoder)
        self.optimizer_dis.setup(self.discriminator)
        self.optimizer_enc.use_cleargrads()
        self.optimizer_dec.use_cleargrads()
        self.optimizer_dis.use_cleargrads()

        # Loss
        self.reconstruction_loss = ReconstructionLoss()
        self.gan_loss = GANLoss()
            
    def train(self, x, z_p):
        # Enc/Dec
        z = self.encoder(x)
        x_rec = self.decoder(z)

        # Dis for x and x_rec
        d_x = self.discriminator(x)
        hiddens_d_x = self.discriminator.hiddens
        d_x_rec = self.discriminator(x_rec)
        hiddens_d_x_rec = self.discriminator.hiddens

        # Reconstruction Loss for Dis
        l_dis = 0
        for h_d_x, h_d_x_rec in zip(hiddens_d_x, hiddens_d_x_rec):
            l_dis += self.reconstruction_loss(h_d_x, h_d_x_rec)
        
        # Generate
        bs = z.shape[0]
        z_p = self._generate(bs)
        x_gen = self.generator(z_p)
        d_x_gen = self.discriminator(x_gen)

        # GAN loss
        l_gan = self.gan_loss(d_x, d_x_rec, d_x_gen)

        # Backward for enc, dec(=gen), and dis
        ## With these update rule,
        ## gradients of GAN Loss do not influence to Enc's parameters update
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
        
    def test(self, x, bs=10):
        """Generate images from data and random seeds
        """
        z = self.encoder(x, test=True)
        x_rec = self.decoder(z, test=True)
        
        z_p = self._generate(bs)
        x_gen = self.generator(z_p, test=True)

        d_x_rec = self.discriminator(x_rec)
        d_x_gen = self.discriminator(x_gen)

        return x_rec, x_gen, d_x_rec, d_x_gen

    def cleargrads(self, ):
        self.encoder.cleargrads()
        self.decoder.cleargrads()
        self.discriminator.cleargrads()

    def _generate(self, bs=64):
        if self.device:
            z = cp.random.uniform(-1, 1, (bs, 100)).astype(np.float32)
            z = cuda.to_gpu(z, self.device)
            z = Variable(z)
        else:
            z = Variable(np.random.uniform(-1, 1, (bs, 100)).astype(np.float32))
        return z

        

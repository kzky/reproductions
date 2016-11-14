"""Models
"""
import numpy as np
import chainer
import chainer.variable as variable
from chainer.functions.activation import lstm
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer.cuda import cupy
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from collections import OrderedDict
import logging
import time
from utils import to_device

class CNNGenrator(Chain):
    
    def __init__(self, batch_size=64, dims=100, test=False, device=None):
        #TODO: now it hard-coded
        super(CNNGenrator, self).__init__(
            linear=L.Linear(dims, 4 * 4 * 1000)
            deconv0=L.Deconvolution2D(1024, 512, ksize=(2, 2), stride=(2, 2)),
            deconv1=L.Deconvolution2D(512, 256, ksize=(2, 2), stride=(2, 2)),
            deconv2=L.Deconvolution2D(256, 128, ksize=(2, 2), stride=(2, 2)),
            deconv3=L.Deconvolution2D(128, 3, ksize=(2, 2), stride=(2, 2)),
            batch_norm0=L.BatchNormalization(0.9),
            batch_norm1=L.BatchNormalization(0.9),
            batch_norm2=L.BatchNormalization(0.9),
            batch_norm3=L.BatchNormalization(0.9),
            batch_norm4=L.BatchNormalization(0.9),
            )

        # Attribute
        self.batch_size = batch_size
        self.dims = dims
        self.test = test
        self.device = device

    def __call__(self, z):
        """Generate samples fooling the discriminator
        """
        # Conv, one reduce
        h = F.reshape(self.linear(z), (4, 4, 1000))
        h = self.batch_norm0(h, self.test)
        h = F.elu(h)

        # Deconv, then expand
        h = self.deconv0(h)
        h = self.batch_norm1(h, self.test)
        h = F.elu(h)
        h = self.deconv1(h)
        h = self.batch_norm2(h, self.test)
        h = F.elu(h)
        h = self.deconv2(h)
        h = self.batch_norm3(h, self.test)
        h = F.elu(h)
        h = self.deconv3(h)
        h = F.tanh(h)
        
        return x
            
class CNNDiscriminator(Chain):
    
    def __init__(self, test=False, device=None):
        super(CNNDiscriminator, self).__init__(
            conv0=L.Convolution2D(3, 128, ksize=(2, 2), stride=(2, 2)),
            conv1=L.Convolution2D(128, 256, ksize=(2, 2), stride=(2, 2)),
            conv2=L.Convolution2D(256, 512, ksize=(2, 2), stride=(2, 2)),
            conv3=L.Convolution2D(512, 1024, ksize=(2, 2), stride=(2, 2)),
            linear0 = L.Linear(None, 512),
            linear1 = L.Linear(None, 256),
            linear2 = L.Linear(None, 128),
            linear3 = L.Linear(None, 1),
            batch_norm0=L.BatchNormalization(0.9),
            batch_norm1=L.BatchNormalization(0.9),
            batch_norm2=L.BatchNormalization(0.9),
            batch_norm3=L.BatchNormalization(0.9),
            batch_norm4=L.BatchNormalization(0.9),
            batch_norm5=L.BatchNormalization(0.9),
            batch_norm6=L.BatchNormalization(0.9),
            )
        
    def __call__(self, x):
        """
        x: Variable
            Note x is scaled over -1 and 1 to comply with tanh
        """
        h = self.conv0(x)
        h = self.batch_norm0(h)
        h = F.leaky_relu(h)
        h = self.conv1(h)
        h = self.batch_norm1(h)
        h = F.leaky_relu(h)
        h = self.conv2(h)
        h = self.batch_norm2(h)
        h = F.leaky_relu(h)
        h = self.conv3(h)
        h = self.batch_norm3(h)
        h = F.leaky_relu(h)
        h = self.linaer0(h)
        h = self.batch_norm4(h)
        h = F.leaky_relu(h)
        h = self.linaer1(h)
        h = self.batch_norm5(h)
        h = F.leaky_relu(h)
        h = self.linaer2(h)
        h = self.batch_norm6(h)
        h = F.leaky_relu(h)
        h = self.linaer3(h)

        return h

class DCGAN(Chain):

    def __init__(self, test=False, device=None):
        self.test = test
        self.generator = CNNGenrator(test, device)
        self.discriminator = CNNDiscriminator(test, device)

    def __call__(self, x=None, z=None):

        if x:  # max log(D(x)) + log(1 - D(G(z)))
            x_ = self.generator(z)
            y = F.log(F.sigmoid(self.discriminator(x_))) \
              + F.log(1 - F.sigmoid(self.discriminator(x_)))

        else: # min log( 1- D(G(z)) )
            x_ = self.generator(z)
            y = F.log(1 - F.sigmoid(self.discriminator(x_)))

        return y
        



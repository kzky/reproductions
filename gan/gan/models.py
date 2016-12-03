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
    """
    """
    def __init__(self, batch_size=64, dims=100, test=False, device=None):
        #TODO: now it hard-coded
        super(CNNGenrator, self).__init__(
            # He initialization by using np.sqrt(2/d_in)
            linear=L.Linear(dims, 1024 * 4 * 4, wscale=np.sqrt(2)),
            deconv0=L.Deconvolution2D(1024, 512, ksize=(4, 4), stride=(2, 2), pad=1,
                                          wscale=np.sqrt(2)),
            deconv1=L.Deconvolution2D(512, 256, ksize=(4, 4), stride=(2, 2), pad=1,
                                          wscale=np.sqrt(2)),
            deconv2=L.Deconvolution2D(256, 128, ksize=(4, 4), stride=(2, 2), pad=1,
                                          wscale=np.sqrt(2)),
            deconv3=L.Deconvolution2D(128, 3, ksize=(4, 4), stride=(2, 2), pad=1,
                                          wscale=np.sqrt(2)),
            batch_norm0=L.BatchNormalization(1024*4*4, 0.9),
            batch_norm1=L.BatchNormalization(512, 0.9),
            batch_norm2=L.BatchNormalization(256, 0.9),
            batch_norm3=L.BatchNormalization(128, 0.9),
            )

        # Attribute
        self.dims = dims
        self.test = test
        self.device = device

    def __call__(self, z):
        """Generate samples fooling the discriminator
        """
        # Project, one reduce
        h = self.linear(z)
        h = self.batch_norm0(h, self.test)
        bs = h.data.shape[0]
        h = F.reshape(h, (bs, 1024, 4, 4))
        h = F.relu(h)

        # Deconv, then expand
        h = self.deconv0(h)
        h = self.batch_norm1(h, self.test)
        h = F.relu(h)
        h = self.deconv1(h)
        h = self.batch_norm2(h, self.test)
        h = F.relu(h)
        h = self.deconv2(h)
        h = self.batch_norm3(h, self.test)
        h = F.relu(h)
        h = self.deconv3(h)
        x = F.tanh(h)
        return x
            
class CNNDiscriminator(Chain):
    
    def __init__(self, test=False, device=None):
        super(CNNDiscriminator, self).__init__(
            conv0=L.Convolution2D(3, 128, ksize=(4, 4), stride=(2, 2), pad=1,
                                      wscale=np.sqrt(2)),
            conv1=L.Convolution2D(128, 256, ksize=(4, 4), stride=(2, 2), pad=1,
                                      wscale=np.sqrt(2)),
            conv2=L.Convolution2D(256, 512, ksize=(4, 4), stride=(2, 2), pad=1,
                                      wscale=np.sqrt(2)),
            conv3=L.Convolution2D(512, 1024, ksize=(4, 4), stride=(2, 2), pad=1,
                                      wscale=np.sqrt(2)),
            linear0 = L.Linear(1024 * 4 * 4, 1, wscale=np.sqrt(2)),
            batch_norm0=L.BatchNormalization(128, 0.9),
            batch_norm1=L.BatchNormalization(256, 0.9),
            batch_norm2=L.BatchNormalization(512, 0.9),
            batch_norm3=L.BatchNormalization(1024, 0.9),
            )
        self.test = test
        
    def __call__(self, x):
        """
        x: Variable
            Note x is scaled over -1 and 1 to comply with tanh
        """
        h = self.conv0(x)
        h = self.batch_norm0(h, self.test)
        h = F.relu(h)
        h = self.conv1(h)
        h = self.batch_norm1(h, self.test)
        h = F.relu(h)
        h = self.conv2(h)
        h = self.batch_norm2(h, self.test)
        h = F.relu(h)
        h = self.conv3(h)
        h = self.batch_norm3(h, self.test)
        h = F.relu(h)
        h = self.linear0(h)
        h = F.sigmoid(h)

        return h

class DCGAN(Chain):

    def __init__(self, batch_size=64, dims=1000, test=False, device=None):
        self.test = test

        super(DCGAN, self).__init__(
            generator = CNNGenrator(batch_size, dims, test, device),
            discriminator = CNNDiscriminator(test, device)
        )

        self.data_model = None
        self.D_z = None

    def __call__(self, z, x=None):

        if x is not None:  # max log(D(x)) + log(1 - D(G(z)))
            x_ = self.generator(z)
            bs_x = x.shape[0]
            bs_x_ = x_.shape[0]
            D_z = self.discriminator(x_)
            loss = F.sum(F.log(self.discriminator(x))) / bs_x \
              + F.sum(F.log(1 - D_z)) / bs_x_
            self.data_model = x_
            self.D_z = D_z
              
        else: # min log( 1- D(G(z)) )
            x_ = self.generator(z)
            bs = x_.shape[0]
            #loss = F.sum(F.log(1 - self.discriminator(x_))) / bs
            loss = F.sum(F.log(self.discriminator(x_)) / bs
            self.data_model = x_
        return loss

    def set_test(self,):
        self.generator.test = True
        self.discriminator.test = True

    def unset_test(self,):
        self.generator.test = False
        self.discriminator.test = False
        


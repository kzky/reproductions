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

from nnabla.initializer import (
    calc_uniform_lim_glorot,
    ConstantInitializer, NormalInitializer, UniformInitializer)


class Unet(object):
    def __init__(self, ):
        pass


    def conv_block(self, h, scope, maps, kernel=(3, 3), stride=(1, 1), bn=False, test=False):
        with nn.parameter_scope(scope):
            h = PF.convolution(h, maps, kernel=kernel, stride=stride, pad=(1, 1))
            h = PF.batch_normalization(h, batch_stat=not test) if bn else h
            h = F.leaky_relu(h)
            h = F.max_pooling(h, (2, 2)) if stride == (1, 1) else h
        return h
    
    
    def deconv_block(self, h, s, scope, maps, kernel=(3, 3), bn=False, test=False): 
        with nn.parameter_scope(scope):
            h = PF.convolution(h, maps, kernel=kernel, stride=(1, 1), pad=(1, 1), name="conv0")
            h = PF.batch_normalization(h, batch_stat=not test) if bn else h
            h = F.leaky_relu(h)
            h = PF.convolution(h, maps, kernel=kernel, stride=(1, 1), pad=(1, 1), name="conv1")
            h = PF.batch_normalization(h, batch_stat=not test) if bn else h
            h = F.leaky_relu(h)
            h = F.unpooling(h, (2, 2))
            assert h.shape[0] == s.shape[0]
            assert h.shape[2:] == s.shape[2:]
            h = F.concatenate(*[h, s], axis=1)
        return h
    
    
    def __call__(self, x, maps=48, stride=(1, 1), bn=False, test=False):
        # First convolution
        h = PF.convolution(x, maps, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name="conv0")
        h = PF.batch_normalization(h, batch_stat=not test) if bn else h
        h0 = F.leaky_relu(h)
        # Encoder
        h1 = self.conv_block(h0, "enc-conv-1", maps, stride=stride)
        h2 = self.conv_block(h1, "enc-conv-2", maps, stride=stride)
        h3 = self.conv_block(h2, "enc-conv-3", maps, stride=stride)
        h4 = self.conv_block(h3, "enc-conv-4", maps, stride=stride)
        h5 = self.conv_block(h4, "enc-conv-5", maps, stride=stride)
        # Bottleneck
        b = PF.convolution(h5, maps, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name="bottle-neck")
        b = F.unpooling(b, (2, 2))
        b = F.concatenate(*[b, h4], axis=1)
        # Decoder
        d0 = self.deconv_block(b, h3, "dec-conv-1", maps * 2)
        d1 = self.deconv_block(d0, h2, "dec-conv-2", maps * 2)
        d2 = self.deconv_block(d1, h1, "dec-conv-3", maps * 2)
        d3 = self.deconv_block(d2, x, "dec-conv-4", maps * 2)
        # Final convolutions
        f = PF.convolution(d3, 64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name="fconv-0")
        f = PF.batch_normalization(f, batch_stat=not test) if bn else h
        f = F.leaky_relu(f)
        f = PF.convolution(f, 32, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name="fconv-1")
        f = PF.batch_normalization(f, batch_stat=not test) if bn else h
        f = F.leaky_relu(f)
        f = PF.convolution(f, 3, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name="fconv-2")
        return f


class REDNetwork(object):
    """RED Network

    We are not exactly sure what the network architecture is since no details in the paper but assume
    from some description.

    Ex 1) pool/unpool is performed at conv1, conv5, conv9 indicates it is performed for every 4 layers
    Ex 2) Final output should be 3 channel, thus we assume the network predicts the residue which is finally added to the input from the Fig. 1
    """

    def __init__(self, layers=15, step_size=2):
        self.layers = layers
        self.step_size = step_size


    def conv_block(self, h, scope, maps, kernel=(3, 3), stride=(1, 1), bn=False, test=False):
        with nn.parameter_scope(scope):
            h = PF.convolution(h, maps, kernel=kernel, stride=stride, pad=(1, 1))
            h = PF.batch_normalization(h, batch_stat=not test) if bn else h
            h = F.relu(h)
        return h
    
    
    def deconv_block(self, h, scope, maps, kernel=(3, 3), stride=(1, 1), bn=False, test=False): 
        with nn.parameter_scope(scope):
            h = PF.deconvolution(h, maps, kernel=kernel, stride=stride, pad=(1, 1))
            h = PF.batch_normalization(h, batch_stat=not test) if bn else h
            h = F.relu(h)
        return h
    
    
    def __call__(self, x, maps=128, bn=False, test=False):
        h = x

        # Encoder
        encodes = []
        for l in range(self.layers):
            if (l + 1) % 4 == 1:
                h = self.conv_block(h, "conv-{:02d}".format(l), maps, kernel=(4, 4), stride=(2, 2))
            else: 
                h = self.conv_block(h, "conv-{:02d}".format(l), maps)
            if (l + 1) % self.step_size == 0:
                encodes.append(h)
        # Decoder
        for l in range(self.layers - 1, -1, -1):
            if (l + 1) % 4 == 1:
                h = self.deconv_block(h, "deconv-{:02d}".format(l), maps, 
                                      kernel=(4, 4), stride=(2, 2))
            else:
                h = self.deconv_block(h, "deconv-{:02d}".format(l), maps)
                
            if (l + 1) % self.step_size == 0:
                e = encodes.pop()
                h = h + e
        # Residue
        r = PF.convolution(h, 3, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
        pred = x + r
        return pred
        
def l2_loss(x, y):
    return F.squared_error(x, y)


def l1_loss(x, y):
    return F.absolute_error(x, y)

def l0_loss(x, y, gamma, eps=1e-8):
    reshape = [1 for _ in range(len(x.shape))]
    return F.pow2(F.absolute_error(x, y) + eps, F.broadcast(F.reshape(gamma, reshape), x.shape))

def get_loss(loss, x_clean, x_noise, x_recon, use_clean, gamma=None):
    if loss == "l2":
        loss_func = l2_loss
    elif loss == "l1":
        loss_func = l1_loss
    elif loss == "l0":
        loss_func = l0_loss
    else:
        raise ValueError("{} is not supported.".format(loss))

    if loss != "l0":
        lossv = F.mean(loss_func(x_recon, x_noise)) \
               if not use_clean else F.mean(loss_func(x_recon, x_clean))
    else:
        lossv = F.mean(loss_func(x_recon, x_noise, gamma)) \
               if not use_clean else F.mean(loss_func(x_recon, x_clean, gamma))
    return lossv

if __name__ == '__main__':
    # Noise2Noise Nework
    b, c, h, w = 4, 3, 256, 256
    x = nn.Variable.from_numpy_array(np.random.randn(b, c, h, w))
    network = Unet()
    pred = network(x, maps=48)
    print(pred.shape)
    
    # Red Network
    b, c, h, w = 4, 3, 256, 256
    x = nn.Variable.from_numpy_array(np.random.randn(b, c, h, w))
    network = REDNetwork(layers=15, step_size=2)
    pred = network(x, maps=128)
    print(pred.shape)

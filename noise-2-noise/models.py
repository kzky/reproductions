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


def conv_block(h, maps, scope, kernel=(3, 3), stride=(1, 1), bn=False, test=False):
    with nn.parameter_scope(scope):
        h = PF.convolution(h, maps, kernel=kernel, stride=stride, pad=(1, 1))
        h = PF.batch_normalization(h, batch_stat=not test) if bn else h
        h = F.leaky_relu(h)
        h = F.max_pooling(h, (2, 2)) if stride == (1, 1) else h
    return h


def deconv_block(h, s, maps, scope, kernel=(3, 3), bn=False, test=False): 
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


def noise2noise_network(x, maps=48, stride=(1, 1), bn=False, test=False):
    # First convolution
    h = PF.convolution(x, maps, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name="conv0")
    h = PF.batch_normalization(h, batch_stat=not test) if bn else h
    h0 = F.leaky_relu(h)
    # Encoder
    h1 = conv_block(h0, maps, "enc-conv-1", stride=stride)
    h2 = conv_block(h1, maps, "enc-conv-2", stride=stride)
    h3 = conv_block(h2, maps, "enc-conv-3", stride=stride)
    h4 = conv_block(h3, maps, "enc-conv-4", stride=stride)
    h5 = conv_block(h4, maps, "enc-conv-5", stride=stride)
    # Bottleneck
    b = PF.convolution(h5, maps, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name="bottle-neck")
    b = F.unpooling(b, (2, 2))
    b = F.concatenate(*[b, h4], axis=1)
    # Decoder
    d0 = deconv_block(b, h3, maps * 2, scope="dec-conv-1")
    d1 = deconv_block(d0, h2, maps * 2, scope="dec-conv-2")
    d2 = deconv_block(d1, h1, maps * 2, scope="dec-conv-3")
    d3 = deconv_block(d2, x, maps * 2, scope="dec-conv-4")
    # Final convolutions
    f = PF.convolution(d3, 64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name="fconv-0")
    f = PF.batch_normalization(f, batch_stat=not test) if bn else h
    f = F.leaky_relu(f)
    f = PF.convolution(f, 32, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name="fconv-1")
    f = PF.batch_normalization(f, batch_stat=not test) if bn else h
    f = F.leaky_relu(f)
    f = PF.convolution(f, 3, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name="fconv-2")
    return f
    
if __name__ == '__main__':
    b, c, h, w = 4, 3, 256, 256
    x = nn.Variable.from_numpy_array(np.random.randn(b, c, h, w))
    pred = noise2noise_network(x, maps=48)
    print(pred.shape)

import os
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


def spectral_normalization_for_conv(w, itr=1, eps=1e-12, test=False):
    w_shape = w.shape
    W_sn = get_parameter_or_create("W_sn", w_shape, ConstantInitializer(0), False)
    if test:
        return W_sn

    d0 = w.shape[0]            # Out
    d1 = np.prod(w.shape[1:])  # In
    w = F.reshape(w, [d0, d1], inplace=False)
    u0 = get_parameter_or_create("singular-vector", [d0], NormalInitializer(), False)
    u = F.reshape(u0, [1, d0])
    # Power method
    for _ in range(itr):
        # v
        v = F.affine(u, w)
        v = F.div2(v, F.pow_scalar(F.sum(F.pow_scalar(v, 2.), keepdims=True) + eps, 0.5))
        v = F.reshape(v, [d1, 1])
        # u
        u = F.affine(w, v)
        u = F.div2(u, F.pow_scalar(F.sum(F.pow_scalar(u, 2.), keepdims=True) + eps, 0.5))
        u = F.reshape(u, [1, d0])
    # Iterate
    u = F.identity(u, outputs=[u0.data])
    u.persistent = True
    # No grad
    u.need_grad = False
    v.need_grad = False
    # Spectral normalization
    wv = F.affine(w, v)
    sigma = F.affine(u, wv)
    w_sn = F.div2(w, sigma)
    w_sn = F.reshape(w_sn, w_shape)
    w_sn = F.identity(w_sn, outputs=[W_sn.data])
    w_sn.persistent = True
    return w_sn


def spectral_normalization_for_affine(w, itr=1, eps=1e-12, input_axis=1, test=False):
    W_sn = get_parameter_or_create("W_sn", w.shape, ConstantInitializer(0), False)
    if test:
        return W_sn

    d0 = np.prod(w.shape[0:input_axis])  # In
    d1 = np.prod(w.shape[input_axis:])   # Out
    u0 = get_parameter_or_create("singular-vector", [d1], NormalInitializer(), False)
    u = F.reshape(u0, [d1, 1])
    # Power method
    for _ in range(itr):
        # v
        v = F.affine(w, u)
        v = F.div2(v, F.pow_scalar(F.sum(F.pow_scalar(v, 2.), keepdims=True) + eps, 0.5))
        v = F.reshape(v, [1, d0])
        # u
        u = F.affine(v, w)
        u = F.div2(u, F.pow_scalar(F.sum(F.pow_scalar(u, 2.), keepdims=True) + eps, 0.5))
        u = F.reshape(u, [d1, 1])
    # Iterate
    u = F.identity(u, outputs=[u0.data])
    u.persistent = True
    # No grad
    u.need_grad = False
    v.need_grad = False
    # Spectral normalization
    wv = F.affine(v, w)
    sigma = F.affine(wv, u)
    sigma = F.broadcast(F.reshape(sigma, [1 for _ in range(len(w.shape))]), w.shape)
    w_sn = F.div2(w, sigma, outputs=[W_sn.data])
    w_sn.persistent = True
    return w_sn


@parametric_function_api("sn_conv")
def convolution(inp, outmaps, kernel,
                pad=None, stride=None, dilation=None, group=1,
                itr=1, 
                w_init=None, b_init=None,
                base_axis=1, fix_parameters=False, rng=None, with_bias=True,
                sn=True, test=False):
                
    """
    N-D Convolution with a bias term.

    For Dilated Convolution (a.k.a. Atrous Convolusion), refer to:

    - Chen et al., DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. https://arxiv.org/abs/1606.00915

    - Yu et al., Multi-Scale Context Aggregation by Dilated Convolutions. https://arxiv.org/abs/1511.07122

    Args:
        inp (~nnabla.Variable): N-D array.
        outmaps (int): Number of convolution kernels (which is equal to the number of output channels). For example, to apply convolution on an input with 16 types of filters, specify 16.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For example, to apply convolution on an image with a 3 (height) by 5 (width) two-dimensional kernel, specify (3,5).
        pad (:obj:`tuple` of :obj:`int`): Padding sizes for dimensions.
        stride (:obj:`tuple` of :obj:`int`): Stride sizes for dimensions.
        dilation (:obj:`tuple` of :obj:`int`): Dilation sizes for dimensions.
        group (int): Number of groups of channels. This makes connections across channels more sparse by grouping connections along map direction.
        itr (int): Number of iteration of the power method.
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight.
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.

    Returns:
        :class:`~nnabla.Variable`: N-D array.

    """
    if w_init is None:
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(inp.shape[base_axis], outmaps, tuple(kernel)), rng=rng)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()
    w = get_parameter_or_create(
        "W", (outmaps, inp.shape[base_axis] / group) + tuple(kernel),
        w_init, not fix_parameters)
    w_sn = spectral_normalization_for_conv(w, itr=itr, test=test) if sn else w
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (outmaps,), b_init, not fix_parameters)
    return F.convolution(inp, w_sn, b, base_axis, pad, stride, dilation, group)
    

@parametric_function_api("sn_affine")
def affine(inp, n_outmaps,
           base_axis=1,
           w_init=None, b_init=None,
           itr=1, 
           fix_parameters=False, rng=None, with_bias=True,
           sn=True, test=False):
    """
    The affine layer, also known as the fully connected layer. Computes

    .. math::
        {\\mathbf y} = {\\mathbf A} {\\mathbf x} + {\\mathbf b}.

    where :math:`{\\mathbf x}, {\\mathbf y}` are the inputs and outputs respectively,
    and :math:`{\\mathbf A}, {\\mathbf b}` are constants.

    Args:
        inp (~nnabla.Variable): Input N-D array with shape (:math:`M_0 \\times \ldots \\times M_{B-1} \\times D_B \\times \ldots \\times D_N`). Dimensions before and after base_axis are flattened as if it is a matrix.
        n_outmaps (:obj:`int` or :obj:`tuple` of :obj:`int`): Number of output neurons per data.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight.
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias.
        itr (int): Number of iteration of the power method.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.

    Returns:
        :class:`~nnabla.Variable`: :math:`(B + 1)`-D array. (:math:`M_0 \\times \ldots \\times M_{B-1} \\times L`)f

    """
    if not hasattr(n_outmaps, '__iter__'):
        n_outmaps = [n_outmaps]
    n_outmaps = list(n_outmaps)
    n_outmap = int(np.prod(n_outmaps))
    if w_init is None:
        inmaps = np.prod(inp.shape[base_axis:])
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(inmaps, n_outmap), rng=rng)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()
    w = get_parameter_or_create(
        "W", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
        w_init, not fix_parameters)
    input_axis = len(inp.shape) - base_axis
    w_sn = spectral_normalization_for_affine(w, itr=itr, input_axis=input_axis, test=test) if sn else w
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", n_outmaps, b_init, not fix_parameters)
    return F.affine(inp, w_sn, b, base_axis)
    

@parametric_function_api("embed")
def embed(inp, n_inputs, n_features, itr=1, fix_parameters=False, sn=True, test=False):
    """Embed.

    Embed slices a matrix/tensor with indexing array/tensor

    Args:
        x(~nnabla.Variable): [Integer] Indices with shape :math:`(I_0, ..., I_N)`
        n_inputs : number of possible inputs, words or vocabraries
        n_features : number of embedding features
        itr (int): Number of iteration of the power method.
        fix_parameters (bool): When set to `True`, the embedding weight matrix
            will not be updated.

    Returns:
        ~nnabla.Variable: Output with shape :math:`(I_0, ..., I_N, W_1, ..., W_M)`
    """
    w = get_parameter_or_create("W", [n_inputs, n_features],
                                UniformInitializer((-np.sqrt(3.), np.sqrt(3))), not fix_parameters)
    w_sn = spectral_normalization_for_affine(w, itr=itr, test=test) if sn else w
    return F.embed(inp, w_sn)
    

def BN(h, test=False):
    """Batch Normalization"""
    return PF.batch_normalization(h, batch_stat=not test)

@parametric_function_api("ccbn")
def CCBN(h, y, n_classes, test=False, fix_parameters=False, sn=True):
    """Categorical Conditional Batch Normaliazation"""
    # Call the batch normalization once
    shape_stat = [1 for _ in h.shape]
    shape_stat[1] = h.shape[1]
    gamma_tmp = nn.Variable.from_numpy_array(np.ones(shape_stat))
    beta_tmp = nn.Variable.from_numpy_array(np.zeros(shape_stat))
    mean = get_parameter_or_create(
        "mean", shape_stat, ConstantInitializer(0), False)
    var = get_parameter_or_create(
        "var", shape_stat, ConstantInitializer(0), False)
    h = F.batch_normalization(h, beta_tmp, gamma_tmp, mean, var, batch_stat=not test)

    # Condition the gamma and beta with the class label
    b, c = h.shape[0:2]
    with nn.parameter_scope("gamma"):
        gamma = embed(y, n_classes, c, sn=sn, test=test)
        gamma = F.reshape(gamma, [b, c] + [1 for _ in range(len(h.shape[2:]))])
        gamma = F.broadcast(gamma, h.shape)
    with nn.parameter_scope("beta"):
        beta = embed(y, n_classes, c, sn=sn, test=test)
        beta = F.reshape(beta, [b, c] + [1 for _ in range(len(h.shape[2:]))])
        beta = F.broadcast(beta, h.shape)
    return gamma * h + beta


def convblock(h, scopename, maps, kernel, pad=(1, 1), stride=(1, 1), upsample=True, test=False):
    with nn.parameter_scope(scopename):
        if upsample:
            h = F.unpooling(h, kernel=(2, 2))
        h = convolution(h, maps, kernel=kernel, pad=pad, stride=stride, with_bias=False, test=test)
        h = PF.batch_normalization(h, batch_stat=not test)
    return h


@parametric_function_api("attn")
def attnblock(h, r=8, fix_parameters=False, sn=True, test=False):
    """Attention block"""
    x = h

    # 1x1 convolutions
    b, c, s0, s1 = h.shape
    c_r = c // r
    assert c_r > 0
    f_x = convolution(h, c_r, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="f", with_bias=False, sn=sn, test=test)
    g_x = convolution(h, c_r, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="g", with_bias=False, sn=sn, test=test)
    h_x = convolution(h, c, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="h", with_bias=False, sn=sn, test=test)

    # Attend 
    attn = F.batch_matmul(f_x.reshape([b, c_r, -1]), g_x.reshape([b, c_r, -1]), transpose_a=True)
    attn = F.softmax(attn, 1)
    h_x = h_x.reshape([b, c, -1])
    o = F.batch_matmul(h_x, attn)
    o = F.reshape(o, [b, c, s0, s1])

    # Shortcut
    gamma = get_parameter_or_create("gamma", [1, 1, 1, 1], ConstantInitializer(0), not fix_parameters)
    y = gamma * o + x
    return y


def resblock_g(h, y, scopename, 
               n_classes, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), 
               upsample=True, test=False, sn=True):
    """Residual block for generator"""
    s = h

    with nn.parameter_scope(scopename):
        # BN -> Relu -> Upsample -> Conv
        with nn.parameter_scope("conv1"):
            h = CCBN(h, y, n_classes, test=test, sn=sn)
            h = F.relu(h, inplace=True)
            if upsample:
                h = F.unpooling(h, kernel=(2, 2))
            h = convolution(h, maps, kernel=kernel, pad=pad, stride=stride, 
                            with_bias=False, sn=sn, test=test)
        
        # BN -> Relu -> Conv
        with nn.parameter_scope("conv2"):
            h = CCBN(h, y, n_classes, test=test, sn=sn)
            h = F.relu(h, inplace=True)
            h = convolution(h, maps, kernel=kernel, pad=pad, stride=stride, 
                            with_bias=False, sn=sn, test=test)
            
        # Shortcut: Upsample -> Conv
        with nn.parameter_scope("shortcut"):
            if upsample:
                s = F.unpooling(s, kernel=(2, 2))
            s = convolution(s, maps, kernel=kernel, pad=pad, stride=stride, 
                            with_bias=False, sn=sn, test=test)
    #return F.add2(h, s, inplace=True)  #TODO: inplace is permittable?
    return F.add2(h, s)


def resblock_d(h, y, scopename,
               n_classes, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), 
               downsample=True, bn=False, test=False, sn=True):
    """Residual block for discriminator"""
    s = h

    with nn.parameter_scope(scopename):
        # BN -> LeakyRelu -> Conv
        with nn.parameter_scope("conv1"):
            h = CCBN(h, y, n_classes, test=test, sn=sn) if bn else h
            h = F.leaky_relu(h, 0.2)
            h = convolution(h, maps, kernel=kernel, pad=pad, stride=stride, 
                            with_bias=False, sn=sn, test=test)
        
        # BN -> LeakyRelu -> Conv -> Downsample
        with nn.parameter_scope("conv2"):
            h = CCBN(h, y, n_classes, test=test, sn=sn) if bn else h
            h = F.leaky_relu(h, 0.2)
            h = convolution(h, maps, kernel=kernel, pad=pad, stride=stride, 
                            with_bias=False, sn=sn, test=test)
            if downsample:
                h = F.average_pooling(h, kernel=(2, 2))
            
        # Shortcut: Conv -> Downsample
        with nn.parameter_scope("shortcut"):
            s = convolution(s, maps, kernel=kernel, pad=pad, stride=stride, 
                            with_bias=False, sn=sn, test=test)
            if downsample:
                s = F.average_pooling(s, kernel=(2, 2))
    #return F.add2(h, s, inplace=True)  #TODO: inplace is permittable?
    return F.add2(h, s)


def generator(z, y, scopename="generator", 
              maps=1024, n_classes=1000, s=4, test=False, sn=True):
    with nn.parameter_scope(scopename):
        # Affine
        h = affine(z, maps * s * s, with_bias=False, sn=sn, test=test)
        h = F.reshape(h, [h.shape[0]] + [maps, s, s])
        # Resblocks
        h = resblock_g(h, y, "block-1", n_classes, maps, test=test, sn=sn)
        h = resblock_g(h, y, "block-2", n_classes, maps // 2, test=test, sn=sn)
        h = resblock_g(h, y, "block-3", n_classes, maps // 4, test=test, sn=sn)
        h = attnblock(h, sn=sn, test=test)
        h = resblock_g(h, y, "block-4", n_classes, maps // 8, test=test, sn=sn)
        h = resblock_g(h, y, "block-5", n_classes, maps // 16, test=test, sn=sn)
        # Last convoltion
        h = CCBN(h, y, n_classes, test=test, sn=sn)
        h = F.relu(h, inplace=True)
        h = convolution(h, 3, kernel=(3, 3), pad=(1, 1), stride=(1, 1), sn=sn, test=test)
        x = F.tanh(h)
    return x


def discriminator(x, y, scopename="discriminator", 
                  maps=64, n_classes=1000, s=4, bn=False, test=False, sn=True):
    with nn.parameter_scope(scopename):
        # Resblocks
        h = resblock_d(x, y, "block-1", n_classes, maps, downsample=False, test=test, sn=sn)
        h = resblock_d(h, y, "block-2", n_classes, maps * 2, test=test, sn=sn)
        h = resblock_d(h, y, "block-3", n_classes, maps * 4, test=test, sn=sn)
        h = attnblock(h, sn=sn, test=test)  # not use attention for discriminator
        h = resblock_d(h, y, "block-4", n_classes, maps * 8, test=test, sn=sn)
        h = resblock_d(h, y, "block-5", n_classes, maps * 16, test=test, sn=sn)
        h = resblock_d(h, y, "block-6", n_classes, maps * 16, test=test, sn=sn)
        # Last affine
        h = CCBN(h, y, n_classes, test=test, sn=sn) if bn else h
        h = F.leaky_relu(h, 0.2)
        h = F.reshape(h, (h.shape[0], -1))
        o0 = affine(h, 1, sn=sn, test=test)
        # Project discriminator
        e = embed(y, n_classes, h.shape[1], name="projection", sn=sn, test=test)
        o1 = F.sum(h * e, axis=1, keepdims=True)
    return o0 + o1


def gan_loss(d_x_fake, d_x_real=None):
    """Hinge loss"""
    if d_x_real is None:
        return -d_x_fake
    return F.maximum_scalar(1 - d_x_real, 0.0) + F.maximum_scalar(1 + d_x_fake, 0.0)
    
    
if __name__ == '__main__':
    b, c, h, w = 4, 3, 128, 128
    latent = 128

    print("Generator shape")
    z = F.randn(shape=[b, latent])
    y = nn.Variable([b])
    y.d = np.random.choice(np.arange(100), b)
    x = generator(z, y)
    print("x.shape = {}".format(x.shape))

    print("Discriminator shape")
    d = discriminator(x, y)
    print("d.shape = {}".format(d.shape))

    # print("Parameters")
    # for n, v in nn.get_parameters(grad_only=False).items():
    #     print(n, v.shape)

    # print("Attention block")
    # b, c, h, w = 4, 32, 128, 128
    # x = nn.Variable([b, c, h//2, w//2])
    # h = attnblock(x)
    # print("h.shape = {}".format(h.shape))
    # nn.clear_parameters()

    
    # print("Spectral Normalization for Conv")
    # o, i, k0, k1 = 8, 8, 16, 16
    # w = nn.Variable([o, i, k0, k1])
    # w.d = np.random.randn(o, i, k0, k1).astype(np.float32)
    # itr = 3
    # np.random.seed(412)
    # #nn.set_auto_forward(True)
    # w_sn = spectral_normalization_for_conv(w, itr=itr)
    # print("w_sn.shape = {}".format(w_sn))
    # def compute_sigma(w):
    #     np.random.seed(412)
    #     u = np.random.randn(o)
    #     w = np.reshape(w, (o, i*k0*k1))
    #     for _ in range(itr):
    #         v = np.dot(u, w)
    #         v = v / np.sqrt(np.sum(v**2) + 1e-12)
    #         u = np.dot(w, v)
    #         u = u / np.sqrt(np.sum(u**2) + 1e-12)
    #     wv = np.dot(w, v)
    #     sigma = np.dot(u, wv)
    #     return sigma
    # w_sn.forward()
    # print(np.allclose(w_sn.d, w.d / compute_sigma(w.d)))
    # nn.clear_parameters()

    # print("Spectral Normalization for Affine")
    # o, i = 16, 8
    # w = nn.Variable([o, i])
    # sigma = spectral_normalization_for_affine(w, itr=2)
    # print("w_sn.shape = {}".format(w_sn))
    # nn.clear_parameters()

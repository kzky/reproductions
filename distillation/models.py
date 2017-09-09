from __future__ import absolute_import
from six.moves import range

import os

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save

def categorical_error(pred, label):
    """
    Compute categorical error given score vectors and labels as
    numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()

def mnist_resnet_prediction(image, net="Teacher", maps=64, test=False):
    """
    Construct ResNet for MNIST.
    """
    image /= 255.0

    def bn(x):
        return PF.batch_normalization(x, batch_stat=not test)

    def res_unit(x, scope):
        C = x.shape[1]
        with nn.parameter_scope(scope):
            with nn.parameter_scope('conv1'):
                h = F.elu(bn(PF.convolution(x, C / 2, (1, 1), with_bias=False)))
            with nn.parameter_scope('conv2'):
                h = F.elu(
                    bn(PF.convolution(h, C / 2, (3, 3), pad=(1, 1), with_bias=False)))
            with nn.parameter_scope('conv3'):
                h = bn(PF.convolution(h, C, (1, 1), with_bias=False))
        return F.elu(F.add2(h, x, inplace=True))

    with nn.parameter_scope(net):
        # Conv1 --> maps x 32 x 32
        with nn.parameter_scope("conv1"):
            c1 = F.elu(
                bn(PF.convolution(image, maps, (3, 3), pad=(3, 3), with_bias=False)))
        # Conv2 --> maps x 16 x 16
        c2 = F.max_pooling(res_unit(c1, "conv2"), (2, 2))
        # Conv3 --> maps x 8 x 8
        c3 = F.max_pooling(res_unit(c2, "conv3"), (2, 2))
        # Conv4 --> maps x 8 x 8
        c4 = res_unit(c3, "conv4")
        # Conv5 --> maps x 4 x 4
        c5 = F.max_pooling(res_unit(c4, "conv5"), (2, 2))
        # Conv5 --> maps x 4 x 4
        c6 = res_unit(c5, "conv6")
        pl = F.average_pooling(c6, (4, 4))
        with nn.parameter_scope("classifier"):
            y = PF.affine(pl, 10)
    return y

def kl_divergence(pred, label):
    elms = - F.softmax(label, axis=1) * F.log(F.softmax(pred, axis=1))
    loss = F.mean(F.sum(elms, axis=1))
    return loss

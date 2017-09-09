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

def mnist_lenet_prediction(image, net="reference", test=False):
    """
    Construct LeNet for MNIST.
    """
    with nn.parameter_scope(net):
        image /= 255.0
        c1 = PF.convolution(image, 16, (5, 5), name='conv1')
        c1 = F.relu(F.max_pooling(c1, (2, 2)), inplace=True)
        c2 = PF.convolution(c1, 16, (5, 5), name='conv2')
        c2 = F.relu(F.max_pooling(c2, (2, 2)), inplace=True)
        c3 = F.relu(PF.affine(c2, 50, name='fc3'), inplace=True)
        c4 = PF.affine(c3, 10, name='fc4')
    return c4

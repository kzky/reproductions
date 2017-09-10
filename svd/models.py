from __future__ import absolute_import
from six.moves import range

import os
from collections import OrderedDict
import numpy as np

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

def mnist_lenet_prediction(image, scope="reference", test=False):
    """
    Construct LeNet for MNIST.
    """
    with nn.parameter_scope(scope):
        image /= 255.0
        c1 = PF.convolution(image, 16, (5, 5), name='conv1')
        c1 = F.relu(F.max_pooling(c1, (2, 2)), inplace=True)
        c2 = PF.convolution(c1, 16, (5, 5), name='conv2')
        c2 = F.relu(F.max_pooling(c2, (2, 2)), inplace=True)
        c3 = F.relu(PF.affine(c2, 50, name='fc3'), inplace=True)
        c4 = PF.affine(c3, 10, name='fc4')
    return c4

def mnist_lenet_prediction_slim(image, scope="slim", rate=0.75, test=False):
    """
    Construct LeNet for MNIST.
    """
    with nn.parameter_scope(scope):
        image /= 255.0
        c1 = PF.convolution(image, 16, (5, 5), name='conv1')
        c1 = F.relu(F.max_pooling(c1, (2, 2)), inplace=True)
        c2 = PF.convolution(c1, 16, (5, 5), name='conv2')
        c2 = F.relu(F.max_pooling(c2, (2, 2)), inplace=True)

        # SVD applied
        inmaps =  np.prod(c2.shape[1:])  # c * h * w
        outmaps0 = 50  # original outmaps
        outmaps1 = reduce_maps(inmaps, outmaps0, rate)
        d0 = F.relu(PF.affine(c2, outmaps1, name='fc-d0'), inplace=True)
        d1 = F.relu(PF.affine(d0, outmaps0, name='fc-d1'), inplace=True)

        c4 = PF.affine(d1, 10, name='fc4')
    return c4

def reduce_maps(inmaps, outmaps, rrate):
    maps = int(rrate * int(1. * inmaps*outmaps / (inmaps+outmaps)))
    return maps

def decompose_network_and_set_params(model_load_path, 
                                     reference, slim, rrate=0.75):
    # Parameters loaded globally, but call here for consistency
    nn.load_parameters(model_load_path)  

    # Decompose
    with nn.parameter_scope(reference):
        trained_params = nn.get_parameters()
    ## original parameter
    W = trained_params["fc3/affine/W"].d
    ## original maps
    inmaps = W.shape[0]
    outmaps0 = W.shape[1]
    ## new maps, R < N*M / (N+M) * rrate
    outmaps1 = reduce_maps(inmaps, outmaps0, rrate)
    ## singular value decomposition
    U, s, V = np.linalg.svd(W, full_matrices=False)
    S = np.diag(s)
    SV = S.dot(V)
    U_approx = U[:, :outmaps1]
    SV_approx = SV[:outmaps1, :outmaps0]

    # Set trained parameters and decomposed parameters
    ## set trained parameters
    with nn.parameter_scope(slim):
        slim_params = nn.get_parameters()
    for n, v in trained_params.items():
        if not n in slim_params.keys():
            continue
        v_slim = slim_params[n]
        v_slim.d = v.d
    ## set decomposed parameters and original bias
    ## a new bias is introcuded due to decomposition
    slim_params["fc-d0/affine/W"].d = U_approx
    slim_params["fc-d1/affine/W"].d = SV_approx
    b = trained_params["fc3/affine/b"]
    slim_params["fc-d1/affine/b"].d = b

    # Clear the parameters of the reference net 
    with nn.parameter_scope(refernece):
        nn.clear_parameters()

def kl_divergence(pred, label):
    elms = - F.softmax(label, axis=1) * F.log(F.softmax(pred, axis=1))
    loss = F.mean(F.sum(elms, axis=1))
    return loss

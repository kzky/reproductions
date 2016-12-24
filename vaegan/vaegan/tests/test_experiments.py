from vaegan.experiments import MLPExperiment, CNNExperiment
import unittest
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

class TestMLPExperiment(unittest.TestCase):
    device = None
    learning_rate = 0.9
    gamma = 1.
    dim = 784
    batch = 16
    
    def test_train(self, ):
        exp = MLPExperiment(
            act=F.relu,
            device=self.device,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
        )

        z_p = np.random.rand(self.batch, self.dim).astype(np.float32)
        x = np.random.rand(self.batch, self.dim).astype(np.float32)
        exp.train(x, z_p)

    def test_test(self, ):
        exp = MLPExperiment(
            act=F.relu,
            device=self.device,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
        )

        x = np.random.rand(self.batch, self.dim).astype(np.float32)
        exp.test(x)

class TestCNNExperiment(unittest.TestCase):
    device = None
    learning_rate = 0.9
    gamma = 1.
    dim = 784
    batch = 16
    
    def test_train(self, ):
        exp = CNNExperiment(
            act=F.relu,
            device=self.device,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
        )

        z_p = np.random.rand(self.batch, self.dim).astype(np.float32)
        x = np.random.rand(self.batch, self.dim)\
                     .reshape(self.batch, 1, 28, 28)\
                     .astype(np.float32)
        exp.train(x, z_p)

    def test_test(self, ):
        exp = CNNExperiment(
            act=F.relu,
            device=self.device,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
        )
 
        x = np.random.rand(self.batch, self.dim)\
                     .reshape(self.batch, 1, 28, 28)\
                     .astype(np.float32)
        exp.test(x)
        

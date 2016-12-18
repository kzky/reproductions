from vaegan.datasets import MNISTDataReader
from vaegan.experiments import MLPExperiment
import numpy as np
import os
from chainer.cuda import cupy as cp
import sys
import time
import chainer.functions as F
from chainer import Variable
from chainer import optimizers, Variable
from utils import to_device
from chainer import serializers
import scipy.io
import cv2
from utils import generate_norm, generate_unif

def main():
    # Settings
    device = None
    learning_rate = 0.9
    gamma = 1.
    dim = 784
    batch = 64
    bs_picked = 10
    n_samples = 70000  # train + test for mnist
    max_epoch = 100
    iter_epoch = int(1.0 * n_samples / batch)
    max_iter = n_samples * iter_epoch

    home = os.environ.get("HOME")
    train_path = os.path.join(home, "datasets/mnist/train.npz")
    test_path = os.path.join(home, "datasets/mnist/test.npz")

    # Experiment for batch
    exp = MLPExperiment(
        act=F.relu,
        device=device,
        learning_rate=learning_rate,
        gamma=gamma,
    )

    # Datareader
    data_reader = MNISTDataReader()

    # Loop
    epoch = 0
    st = time.time()
    utime = int(st)
    dpath0 = "./MLP_MINST_{}".format(utime)
    os.mkdir(dpath0)
    for i in range(max_iter):
        # Train
        x = data_reader.get_train_batch()  # (-1, 1)
        z_p = generate_unif(batch, dim=100, device=device)
        exp.train(x, z_p)
        
        # Test
        if i % iter_epoch == 0:
            # Generate
            epoch += 1
            bs = x.shape[0]
            x_ = x[np.random.choice(bs, bs_picked, replace=False)]
            ret_val = exp.test(x_, bs=bs_picked)
            x_rec, x_gen, d_x_rec, d_x_gen = ret_val

            # Log
            et = time.time()
            msg = "Epoch:{},ElapsedTime:{},D(x_rec):{},D(x_gen):{}".format(
                epoch, et - st,
                cuda.to_cpu(d_x_rec[0:5]),
                cuda.to_cpu(d_x_gen[0:5])
            )
            print(msg)

            # Save images
            x_ori = (x_ * 127.5 + 127.5).reshape(bs, 28, 28)
            x_rec = (cuda.to_cpu(x_rec.data) * 127.5 + 127.5).reshape(bs, 28, 28)
            x_gen = (cuda.to_cpu(x_gen.data) * 127.5 + 127.5).reshape(bs, 28, 28)
            dpath1 = os.path.join(dpath0, "{:05d}".format(epoch))
            os.mkdir(dpath1)
            for i, img in enumerate(imgs):
                fpath = os.path.join(dpath1, "ori_{:05}.png".format(i))
                cv2.imwrite(fpath, x_ori)
                fpath = os.path.join(dpath1, "rec_{:05}.png".format(i))
                cv2.imwrite(fpath, x_rec)
                fpath = os.path.join(dpath1, "gen_{:05}.png".format(i))
                cv2.imwrite(fpath, x_gen)

            # Save model
            fpath = os.path.join(dpath0, "MLP_MINST_{}.model".format(int(et)))
            serializers.save_hdf5(fpath, model)

            st = time.time()

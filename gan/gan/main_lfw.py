from gan.datasets import LFWDataReader, Separator
import numpy as np
import os
from chainer.cuda import cupy
import sys
import time
import chainer.functions as F
from chainer import Variable
from chainer import optimizers, Variable
from utils import to_device
import serializers

def main():
    # Settings
    device = int(sys.argv[1]) if len(sys.argv) > 1 else None
    xp = numpy if device else cupy
    batch_size = 64
    n_train_data = 13233
    k_steps = 1

    learning_rate = 1. * 1e-3
    n_epoch = 100
    decay = 0.9
    iter_epoch = n_train_data / batch_size
    n_iter = n_epoch * iter_epoch
    
    home = os.environ.get("HOME")
    fpath = os.path.join(home, "datasets/lfw")
    data_reader = LFWDataReader(fpath, batch_size)

    # Model
    model = DCGAN()
    model.generator.to_gpu() if device else None
    model.discriminator.to_gpu() if device else None

    # Optimizers
    optimizer_gen = optimizers.Adam(learning_rate)
    optimizer_gen.setup(model.generator)
    optimizer_gen.use_cleargrads()
    optimizer_dis = optimizers.Adam(learning_rate)
    optimizer_dis.setup(model.discriminator)
    optimizer_dis.use_cleargrads()
    
    # Train
    for i in range(n_iter):
        
        # Maximize Discriminator-related objective
        for k in range(k_steps):
            x_data = data_reader.get_train_batch()
            x = Variable(to_device(x_data, device))
            z = Variable(to_device(xp.random.rand(batch_size, 100), device))
            l = -1.0 *  model(z, x)
            optimizer_dis.update()

        # Minimize Generator-related objective
        z = Variable(to_device(xp.random.rand(batch_size, 100), device))
        l = model(z)
        optimizer_gen.update()

        # Eval
        if (i + 1) % iter_epoch == 0:
            # Generate image and save
            z = Variable(to_device(xp.random.rand(batch_size, 100), device))
            x = model.generator(z)
            scipy.io.savemat({"x":x.data})

            # Save model
            utime = int(time.time())
            model_name = "LFWGAN_{}.model".fromat(utime)
            serializers.save_hdf5(model_name, model)

if __name__ == '__main__':
    main()

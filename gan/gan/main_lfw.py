from gan.datasets import LFWDataReader
from gan.models import DCGAN
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

def main():
    # Settings
    device = int(sys.argv[1]) if len(sys.argv) > 1 else None
    xp = np if device else cp
    batch_size = 100
    test = False
    n_train_data = 13233
    k_steps = 1

    learning_rate = 2. * 1e-4
    beta = 0.5
    n_epoch = 1000
    decay = 0.9
    iter_epoch = n_train_data / batch_size
    n_iter = n_epoch * iter_epoch
    
    home = os.environ.get("HOME")
    fpath = os.path.join(home, "datasets/lfw")
    data_reader = LFWDataReader(fpath, batch_size)

    # Model
    model = DCGAN(batch_size=64, dims=100, test=test, device=device)
    model.to_gpu(device) if device else None

    # Optimizers
    optimizer_gen = optimizers.Adam(learning_rate, beta)
    optimizer_gen.setup(model.generator)
    optimizer_gen.use_cleargrads()
    optimizer_dis = optimizers.Adam(learning_rate, beta)
    optimizer_dis.setup(model.discriminator)
    optimizer_dis.use_cleargrads()
    
    # Train
    epoch = 0
    st = time.time()
    utime = int(st)
    dpath0 = "./LFWGAN_{}".format(utime)
    os.mkdir(dpath0)
    for i in range(n_iter):
        
        # Maximize Discriminator-related objective
        for k in range(k_steps):
            x_data = data_reader.get_train_batch()
            x = Variable(to_device(x_data, device))
            bs = x_data.shape[0]
            z = Variable(to_device(np.random.uniform(-1, 1, (batch_size, 100)).astype(np.float32), device))
            l = -model(z, x)
            model.cleargrads()
            l.backward()
            optimizer_dis.update()

        # Minimize Generator-related objective
        z = Variable(to_device(np.random.uniform(-1, 1, (batch_size, 100)).astype(np.float32), device))
        l = model(z)
        model.cleargrads()
        l.backward()
        optimizer_gen.update()

        # Eval
        if (i+1) % iter_epoch == 0:
            model.set_test()
            et = int(time.time())
            
            # Generate image
            z = Variable(to_device(
                np.random.uniform(-1, 1, (batch_size, 100)).astype(np.float32), device))
            l = model(z, x)
            msg = "ElapsedTime:{},Epoch:{},Loss:{},D_z:{}".format(
                et - st, epoch, l.data, model.D_z.data[0:5])
            print(msg)

            # Save images
            data = (to_device(model.data_model.data) * 127.5) + 127.5
            imgs = data.transpose(0, 2, 3, 1)
            dpath1 = os.path.join(dpath0, str(et))
            os.mkdir(dpath1)
            for i, img in enumerate(imgs):
                fpath = os.path.join(dpath1, "{:05}.png".format(i))
                cv2.imwrite(fpath, img)

            # Save model
            fpath = os.path.join(dpath0, "LFWGAN_{}.model".format(int(et)))
            serializers.save_hdf5(fpath, model)

            model.unset_test()
            epoch += 1
            st = time.time()

if __name__ == '__main__':
    main()

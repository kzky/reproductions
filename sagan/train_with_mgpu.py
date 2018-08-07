import os
import numpy as np
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.communicators as C
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
import nnabla.utils.save as save
from nnabla.ext_utils import get_extension_context
from args import get_args, save_args

from helpers import MonitorImageTileWithName, generate_random_class
from models import generator, discriminator, gan_loss
from imagenet_data import data_iterator_imagenet

def train(args):
    # Communicator and Context
    extension_module = "cudnn"
    ctx = get_extension_context(extension_module)
    comm = C.MultiProcessDataParalellCommunicator(ctx)
    comm.init()
    n_devices = comm.size
    mpi_rank = comm.rank
    device_id = comm.local_rank
    ctx.device_id = str(device_id)
    nn.set_default_context(ctx)

    # Model (not use test mode when training either generator and discrimiantor)
    np.random.seed(412)  # workaround to start with the same weights in the distributed system.
    # generator loss
    z = F.randn(0, 1.0, [args.batch_size, args.latent])
    y = nn.Variable([args.batch_size], need_grad=False)
    x_fake = generator(z, y, maps=args.maps, sn=args.not_sn).apply(persistent=True)
    d_fake = discriminator(x_fake, y, maps=args.maps // 16, sn=args.not_sn)
    loss_gen = gan_loss(d_fake)
    # discriminator loss
    x_real = nn.Variable([args.batch_size, 3, args.image_size, args.image_size], need_grad=False)
    x_augm = F.flip(x_real, (2, ))
    d_real = discriminator(x_augm, y, maps=args.maps // 16, sn=args.not_sn)
    loss_dis = gan_loss(d_fake, d_real)
    # generator with fixed value for test
    z_test = nn.Variable.from_numpy_array(np.random.randn(args.batch_size, args.latent))
    y_test = nn.Variable.from_numpy_array(generate_random_class(args.n_classes, args.batch_size))
    x_test = generator(z_test, y_test, maps=args.maps, test=True, sn=args.not_sn)
    
    # Solver
    solver_gen = S.Adam(args.lrg, args.beta1, args.beta2)
    solver_dis = S.Adam(args.lrd, args.beta1, args.beta2)
    with nn.parameter_scope("generator"):
        solver_gen.set_parameters(nn.get_parameters())
    with nn.parameter_scope("discriminator"):
        solver_dis.set_parameters(nn.get_parameters())

        # Monitor
    if comm.rank == 0:
        monitor = Monitor(args.monitor_path)
        monitor_loss_gen = MonitorSeries("Generator Loss", monitor, interval=10)
        monitor_loss_dis = MonitorSeries("Discriminator Loss", monitor, interval=10)
        monitor_time = MonitorTimeElapsed("Training Time", monitor, interval=10)
        monitor_image_tile_train = MonitorImageTileWithName("Image Tile Train", monitor,
                                                            num_images=args.batch_size,
                                                            normalize_method=lambda x: (x + 1.) / 2.)
        monitor_image_tile_test = MonitorImageTileWithName("Image Tile Test", monitor,
                                                            num_images=args.batch_size,
                                                            normalize_method=lambda x: (x + 1.) / 2.)
    # DataIterator
    rng = np.random.RandomState(410)
    di_train = data_iterator_imagenet(args.batch_size, args.train_cachefile_dir, rng=rng)

    # Train loop
    normalize_method = lambda x: (x - 127.5) / 127.5
    for i in range(args.max_iter):
        # Train discriminator
        x_fake.need_grad = False  # no need for discriminator backward
        solver_dis.zero_grad()
        for _ in range(args.accum_grad):
            x_data, y_data = di_train.next()
            x_real.d, y.d = normalize_method(x_data), y_data.flatten()
            loss_dis.forward(clear_no_need_grad=True)
            loss_dis.backward(1.0 / (args.accum_grad * n_devices), clear_buffer=True)
        with nn.parameter_scope("discriminator"):
            comm.all_reduce([v.grad for v in nn.get_parameters().values()])
        solver_dis.update()

        # Train genrator
        if (i + 1) % args.n_critc == 0:
            x_fake.need_grad = True  # need for generator backward
            solver_gen.zero_grad()
            for _ in range(args.accum_grad):
                y_data = generate_random_class(args.n_classes, args.batch_size)
                y.d = y_data
                loss_gen.forward(clear_no_need_grad=True)
                loss_gen.backward(1.0 / (args.accum_grad * n_devices), clear_buffer=True)
            with nn.parameter_scope("generator"):
                comm.all_reduce([v.grad for v in nn.get_parameters().values()])
            solver_gen.update()

        # Synchronize by averaging the weights over devices using allreduce
        # if i % args.sync_weight_every_itr == 0:
        #     weights = [x.data for x in nn.get_parameters().values()]
        #     comm.all_reduce(weights, division=True, inplace=True)

        # Save model and image
        if i % args.save_interval == 0 and comm.rank == 0:
            x_test.forward(clear_buffer=True)
            nn.save_parameters(os.path.join(args.monitor_path, "params_{}.h5".format(i)))
            monitor_image_tile_train.add("image_{}".format(i), x_fake.d)
            monitor_image_tile_test.add("image_{}".format(i), x_test.d)

        # Monitor
        if comm.rank == 0:
            monitor_loss_gen.add(i, loss_gen.d.copy())
            monitor_loss_dis.add(i, loss_dis.d.copy())
            monitor_time.add(i)

    if comm.rank == 0:
        x_test.forward(clear_buffer=True)
        nn.save_parameters(os.path.join(args.monitor_path, "params_{}.h5".format(i)))
        monitor_image_tile_train.add("image_{}".format(i), x_fake.d)
        monitor_image_tile_test.add("image_{}".format(i), x_test.d)

def main():
    args = get_args()
    save_args(args)

    train(args)

if __name__ == '__main__':
    main() 

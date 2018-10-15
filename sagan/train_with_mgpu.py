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

    # Args
    latent = args.latent
    maps = args.maps
    batch_size = args.batch_size
    image_size = args.image_size
    n_classes = args.n_classes
    not_sn = args.not_sn
    
    # Model
    np.random.seed(412)  # workaround to start with the same weights in the distributed system.
    # generator loss
    z = nn.Variable([batch_size, latent])
    y_fake = nn.Variable([batch_size])
    x_fake = generator(z, y_fake, maps=maps, n_classes=n_classes, sn=not_sn).apply(persistent=True)
    p_fake = discriminator(x_fake, y_fake, maps=maps // 16, n_classes=n_classes, sn=not_sn)
    loss_gen = gan_loss(p_fake)
    # discriminator loss
    y_real = nn.Variable([batch_size])
    x_real = nn.Variable([batch_size, 3, image_size, image_size])
    p_real = discriminator(x_real, y_real, maps=maps // 16, n_classes=n_classes, sn=not_sn)
    loss_dis = gan_loss(p_fake, p_real)
    # generator with fixed value for test
    z_test = nn.Variable.from_numpy_array(np.random.randn(batch_size, latent))
    y_test = nn.Variable.from_numpy_array(generate_random_class(n_classes, batch_size))
    x_test = generator(z_test, y_test, maps=maps, n_classes=n_classes, test=True, sn=not_sn)
                       
    # Solver
    solver_gen = S.Adam(args.lrg, args.beta1, args.beta2)
    solver_dis = S.Adam(args.lrd, args.beta1, args.beta2)
    with nn.parameter_scope("generator"):
        params_gen = nn.get_parameters()
        solver_gen.set_parameters(params_gen)
    with nn.parameter_scope("discriminator"):
        params_dis = nn.get_parameters()
        solver_dis.set_parameters(params_dis)

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
    rng = np.random.RandomState(device_id)
    di_train = data_iterator_imagenet(args.train_dir, args.dirname_to_label_path,
                                      args.batch_size, n_classes=args.n_classes, 
                                      rng=rng)

    # Train loop
    for i in range(args.max_iter):
        # Train discriminator
        x_fake.need_grad = False  # no need for discriminator backward
        solver_dis.zero_grad()
        for _ in range(args.accum_grad):
            # feed x_real and y_real
            x_data, y_data = di_train.next()
            x_real.d, y_real.d = x_data, y_data.flatten()
            # feed z and y_fake
            z_data = np.random.randn(args.batch_size, args.latent)
            y_data = generate_random_class(args.n_classes, args.batch_size)
            z.d, y_fake.d = z_data, y_data
            loss_dis.forward(clear_no_need_grad=True)
            loss_dis.backward(1.0 / (args.accum_grad * n_devices), clear_buffer=True)
        comm.all_reduce([v.grad for v in params_dis.values()])
        solver_dis.update()

        # Train genrator
        if (i + 1) % args.n_critic == 0:
            x_fake.need_grad = True  # need for generator backward
            solver_gen.zero_grad()
            for _ in range(args.accum_grad):
                z_data = np.random.randn(args.batch_size, args.latent)
                y_data = generate_random_class(args.n_classes, args.batch_size)
                z.d, y_fake.d = z_data, y_data
                loss_gen.forward(clear_no_need_grad=True)
                loss_gen.backward(1.0 / (args.accum_grad * n_devices), clear_buffer=True)
            comm.all_reduce([v.grad for v in params_gen.values()])
            solver_gen.update()

        # Synchronize by averaging the weights over devices using allreduce
        if i % args.sync_weight_every_itr == 0:
            weights = [v.data for v in nn.get_parameters().values()]
            comm.all_reduce(weights, division=True, inplace=True)

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

import os
import numpy as np
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
import nnabla.utils.save as save
from nnabla.ext_utils import get_extension_context
from args import get_args, save_args

from helpers import MonitorImageTileWithName
from models import generator, discriminator, gan_loss
from imagenet_data import data_iterator_imagenet

def train(args):
    # Context
    ctx = get_extention_context(args.context)
    nn.set_default_context(ctx)

    # Model
    z = F.randn([args.batch_size, args.latent])    
    y = nn.Variable([args.batch_size], need_grad=False)
    x_real = nn.Variable([args.batch_size, 3, args.image_size, args.image_size], need_grad=False)
    x_fake = generator(z, y, maps=args.latent)
    x_fake.persistent = True
    d_fake = discriminator(x_fake, y)
    d_real = discriminator(x_real, y)
    loss_gen = F.mean(gan_loss(d_fake))
    loss_dis = F.mean(gan_loss(d_fake, d_real))
    
    z_test = nn.Variable.from_numpy_array(np.random.randn(args.batch_size, args.latent))
    y_test = nn.Variable.from_numpy_array(
        np.random.choice(np.arange()replace=True).reshape([args.batch_size, args.latent]))
    x_test = generator(z_test, y_test, maps=args.latent)
    
    # Solver
    solver_gen = S.Adam(args.learning_rage_for_generator, args.beta1, args.beta2)
    solver_dis = S.Adam(args.learning_rage_for_discriminator, args.beta1, args.beta2)
    with nn.parameter_scope("generator"):
        solver_gen.set_parameters(nn.get_parameters())
    with nn.parameter_scope("discriminator"):
        solver_dis.set_parameters(nn.get_parameters())

    # Monitor
    monitor = Monitor(args.monitor_path)
    monitor_loss_gen = MonitorSeries("Generator Loss", monitor, interval=10)
    monitor_loss_dis = MonitorSeries("Discriminator Loss", monitor, interval=10)
    monitor_time = MonitorTimeElapsed(
        "Training Time per Resolution", monitor, interval=1)
    monitor_image_tile = MonitorImageTileWithName("Image Tile", monitor,
                                                  num_images=4,
                                                  normalize_method=lambda x: (x + 1.) / 2.)
    # DataIterator
    rng = np.random.RandomState(410)
    di_train = data_iterator_imagenet(args.batch_size, args.train_cachefile_dir, rng=rng)
    di_test = data_iterator_imagenet(args.batch_size, args.val_cachefile_dir, rng=False)
    
    # Train loop
    for i in range(args.max_iter):
        # Feed data
        x_real.d, y.d = di_train.next()
        
        # Train genrator
        loss_gen.forward(clear_no_need_grad=True)
        solver_gen.zero_grad()
        loss_gen.backward(clear_buffer=True)
        solver_gen.update()
        
        # Train discriminator
        loss_dis.forward(clear_no_need_grad=True)
        solver_dis.zero_grad()
        loss_dis.backward(clear_buffer=True)
        solver_gen.update()
        
        # Save model and image
        if i % args.save_interval == 0:
            x_test.forward(clear_buffer=True)
            nn.save_parameters(os.path.join(args.monitor_path, "params_{}.h5".format(i)))
            monitor_image_tile.add("image_{}".format(i), x_test.d)

    x_test.forward(clear_buffer=True)
    nn.save_parameters(os.path.join(args.monitor_path, "params_{}.h5".format(i)))
    monitor_image_tile.add("image_{}".format(i))

def main():
    args = get_args()
    save_args(args)

    train(args)

if __name__ == '__main__':
    main() 

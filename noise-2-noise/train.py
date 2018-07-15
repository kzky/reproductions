import os
import numpy as np
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.communicators as C
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed, MonitorImageTile
from nnabla.ext_utils import get_extension_context
import nnabla.utils.save as save

from helpers import apply_noise
from datasets import data_iterator_imagenet
from args import get_args, save_args
from models import REDNetwork, Noise2NoiseNetwork, get_loss

def train(args):
    # Communicator and Context
    extension_module = args.context
    ctx = get_extension_context(extension_module)
    nn.set_default_context(ctx)

    # Model
    if args.net == "RED":
        net = REDNetwork(layers=30, step_size=2)
    elif args.net == "noise2noise":
        net = Noise2NoiseNetwork()
    x = nn.Variable([args.batch_size, 3, args.ih, args.iw])
    x.persistent = True
    x_noise = nn.Variable([args.batch_size, 3, args.ih, args.iw])
    x_noise.persistent = True
    x_recon = net(x_noise)
    x_recon.persistent = True
    loss = F.mean(get_loss(args.loss)(x_recon, x))
    
    # Solver
    solver = S.Adam(args.lr, args.beta1, args.beta2)
    solver.set_parameters(nn.get_parameters())

    # Monitor
    def normalize_method(x):
        max_idx = np.where(x > 255)
        min_idx = np.where(x < 0)
        x[max_idx] = 255
        x[min_idx] = 0
        return x
    monitor = Monitor(args.monitor_path)
    monitor_loss = MonitorSeries("Reconstruction Loss", monitor, interval=10)
    monitor_time = MonitorTimeElapsed("Training Time per Resolution", monitor, interval=10)
    monitor_image_train_clean = MonitorImageTile("Image Tile Train Clean", monitor,
                                                 num_images=args.batch_size, 
                                                 normalize_method=normalize_method, 
                                                 interval=10)
    monitor_image_train_noisy = MonitorImageTile("Image Tile Train Noisy", monitor,
                                                 num_images=args.batch_size, 
                                                 normalize_method=normalize_method, 
                                                 interval=10)
    monitor_image_train_recon = MonitorImageTile("Image Tile Train Recon", monitor,
                                                 num_images=args.batch_size, 
                                                 normalize_method=normalize_method, 
                                                 interval=10)
    # DataIterator
    rng = np.random.RandomState(410)
    di = data_iterator_imagenet(args.train_data_path, args.batch_size, rng=rng)
    
    # Train loop
    for i in range(args.max_iter):
        # Data feed
        x_data, _ = di.next()
        x.d = x_data
        x_noise.d = apply_noise(x_data, args.noise_level, distribution=args.dist)

        # Forward, backward, and update
        loss.forward(clear_no_need_grad=True)
        solver.zero_grad()
        loss.backward(clear_buffer=True)
        solver.update()

        # Save model and images
        if i % args.save_interval == 0:
            nn.save_parameters("{}/param_{}.h5".format(args.monitor_path, i))
            monitor_image_train_clean.add(i, x_data)
            monitor_image_train_noisy.add(i, x_noise.d)
            monitor_image_train_recon.add(i, x_recon.d)
        
        # Monitor
        monitor_loss.add(i, loss.d)
        monitor_time.add(i)

def main():
    args = get_args()
    save_args(args)

    train(args)

if __name__ == '__main__':
    main() 

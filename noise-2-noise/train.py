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
from helpers import RampdownLRScheduler
from datasets import data_iterator_imagenet
from args import get_args, save_args
from models import REDNetwork, Unet, get_loss

def train(args):
    # Communicator and Context
    extension_module = args.context
    ctx = get_extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)

    # Model
    if args.net == "red30":
        net = REDNetwork(layers=30, step_size=2)
    elif args.net == "unet":
        net = Unet()
    x_clean = nn.Variable([args.batch_size_per_replica * args.n_replica, 3, args.ih, args.iw])
    x_clean.persistent = True
    x_noise = nn.Variable([args.batch_size_per_replica * args.n_replica, 3, args.ih, args.iw])
    x_noise.persistent = True
    x_noise_t = nn.Variable([args.batch_size_per_replica * args.n_replica, 3, args.ih, args.iw])
    x_recon = net(x_noise)
    x_recon.persistent = True
    gamma = nn.Variable.from_numpy_array(np.asarray([2.]))
    mask = nn.Variable.from_numpy_array(np.ones(x_recon.shape))
    loss = get_loss(args.loss, x_recon, x_clean, x_noise_t, args.use_clean, mask, gamma)

    # Solver
    solver = S.Adam(args.lr, args.beta1, args.beta2)
    solver.set_parameters(nn.get_parameters())
    lr_scheduler = RampdownLRScheduler(solver)

    # Monitor
    def normalize_method(x):
        max_idx = np.where(x > 255)
        min_idx = np.where(x < 0)
        x[max_idx] = 255
        x[min_idx] = 0
        return x
    monitor = Monitor(args.monitor_path)
    monitor_loss = MonitorSeries("Reconstruction Loss", monitor, interval=10)
    monitor_time = MonitorTimeElapsed("Training Time", monitor, interval=10)
    monitor_image_clean = MonitorImageTile("Image Tile Train Clean",
                                           monitor,
                                           num_images=1, 
                                           normalize_method=normalize_method, 
                                           interval=1)
    monitor_image_noisy = MonitorImageTile("Image Tile Train Noisy",
                                           monitor,
                                           num_images=1,
                                           normalize_method=normalize_method, 
                                           interval=1)
    monitor_image_recon = MonitorImageTile("Image Tile Train Recon",
                                           monitor,
                                           num_images=1,
                                           normalize_method=normalize_method, 
                                           interval=1)
    monitor_image_target = MonitorImageTile("Image Tile Train Target",
                                           monitor,
                                           num_images=1,
                                           normalize_method=normalize_method, 
                                           interval=1)
    # DataIterator
    rng = np.random.RandomState(410)
    di = data_iterator_imagenet(args.train_data_path, args.batch_size_per_replica, rng=rng)
    
    # Train loop
    for i in range(args.max_iter):
        # Data feed
        x_data, _ = di.next()
        if args.use_clean:
            assert args.n_replica == 1
            x_clean.d = x_data
        x_noise.d, x_noise_t.d, noise = apply_noise(x_data, 
                                                    args.n_replica,
                                                    args.noise_level, 
                                                    distribution=args.noise_dist)

        # Forward, backward, and update
        scale = noise if args.noise_dist == "bernoulli" else np.ones(x_recon.shape)
        mask.d = scale
        loss.forward(clear_no_need_grad=True)
        solver.zero_grad()
        loss.backward(clear_buffer=True)
        solver.update()

        # Schedule LR
        lr_scheduler(loss.d)

        # Update gamma if L0 loss is used
        gamma.d = gamma.d.copy() * (1.0 - 1.0 * i / args.max_iter) if args.loss == "l0" else 2.0

        # Save model and images
        if i % args.save_interval == 0:
            nn.save_parameters("{}/param_{}.h5".format(args.monitor_path, i))
            monitor_image_clean.add(i, x_data)
            monitor_image_noisy.add(i, x_noise.d)
            monitor_image_recon.add(i, x_recon.d)
            monitor_image_target.add(i, x_noise_t.d)
        
        # Monitor
        monitor_loss.add(i, loss.d)
        monitor_time.add(i)

def main():
    args = get_args()
    save_args(args)

    train(args)

if __name__ == '__main__':
    main() 

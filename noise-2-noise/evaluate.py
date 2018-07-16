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


def evaluate(args):
    # Model
    if args.net == "RED":
        net = REDNetwork(layers=30, step_size=2)
    elif args.net == "noise2noise":
        net = Noise2NoiseNetwork()
    nn.load_parameters(args.model_load_path)

    # Data iterator
    if args.val_dataset == "kodak":
        ds = KodacDataSource()
        di = data_iterator_kodak(ds)
    elif args.val_dataset == "bsds300":
        ds = BSDSDataSource("bsds300")
        di = data_iterator(ds)
    elif args.val_dataset == "bsds500":
        ds = BSDSDataSource("bsds500")
        di = data_iterator(ds)
    elif args.val_dataset == "set5":
        ds = BSDSDataSource("set5")
        di = data_iterator(ds)
    elif args.val_dataset == "set14":
        ds = BSDSDataSource("set14")
        di = data_iterator(ds)
    else:
        raise ValueError("{} is not supported.".format(args.val_dataset))

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
    monitor_image_train_clean = MonitorImage("Image Train Clean", monitor,
                                                 num_images=1, 
                                                 normalize_method=normalize_method, 
                                                 interval=1)
    monitor_image_train_noisy = MonitorImage("Image Train Noisy", monitor,
                                                 num_images=1, 
                                                 normalize_method=normalize_method, 
                                                 interval=1)
    monitor_image_train_recon = MonitorImage("Image Train Recon", monitor,
                                                 num_images=1, 
                                                 normalize_method=normalize_method, 
                                                 interval=1)

    # Evaluate
    for i in range(ds.size):
        # Read data
        x_data, _ = di.next()
        
        # Create model
        x = nn.Variable([1, 3, x_data.shape[2], x_data.shape[3]])
        x.persistent = True
        x_noise = nn.Variable([1, 3, x_data.shape[2], x_data.shape[3]])
        x_noise.persistent = True
        x_recon = net(x_noise)
        x_recon.persistent = True
        x.d = x_data
        x_noise.d = apply_noise(x_data, args.noise_level, distribution=args.dist, fix=True)

        # Forward (denoise)
        loss.forward(clear_buffer=True)

        # Save
        monitor_image_train_clean.add(i, x_data)
        monitor_image_train_noisy.add(i, x_noise.d)
        monitor_image_train_recon.add(i, x_recon.d)

        # Clear memory since the input is varaible size.
        import nnabla_ext.cuda
        nnabla_ext.cuda.clear_memory_cache()


def main():
    args = get_args()
    save_args(args)

    evaluate(args)


if __name__ == '__main__':
    main()

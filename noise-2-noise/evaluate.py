import os
import numpy as np
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.communicators as C
from nnabla.utils.data_iterator import data_iterator
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed, MonitorImage
from nnabla.ext_utils import get_extension_context
import nnabla.utils.save as save


from helpers import apply_noise, psnr
from datasets import KodakDataSource, BSDSDataSource, SetDataSource
from args import get_args, save_args
from models import REDNetwork, Unet, get_loss


def evaluate(args):
    # Context
    ctx = get_extension_context(args.context, device_id=args.device_id)
    nn.set_default_context(ctx)

    # Model
    if args.net == "RED":
        net = REDNetwork(layers=30, step_size=2)
    elif args.net == "unet":
        net = Unet()
    nn.load_parameters(args.model_load_path)


    # Data iterator
    if args.val_dataset == "kodak":
        ds = KodakDataSource()
        di = data_iterator(ds, batch_size=1)
    elif args.val_dataset == "bsds300":
        ds = BSDSDataSource("bsds300")
        di = data_iterator(ds, batch_size=1)
    elif args.val_dataset == "bsds500":
        ds = BSDSDataSource("bsds500")
        di = data_iterator(ds, batch_size=1)
    elif args.val_dataset == "set5":
        ds = BSDSDataSource("set5")
        di = data_iterator(ds, batch_size=1)
    elif args.val_dataset == "set14":
        ds = BSDSDataSource("set14")
        di = data_iterator(ds, batch_size=1)
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
    monitor_loss = MonitorSeries("{} Evaluation Metric {}".format(args.val_dataset, 
                                                                  args.noise_dist), 
                                 monitor, interval=1)
    monitor_image_test_clean = MonitorImage("{} Image Test {} Clean".format(args.val_dataset, 
                                                                            args.noise_dist), 
                                            monitor,
                                            num_images=1, 
                                            normalize_method=normalize_method, 
                                            interval=1)
    monitor_image_test_noisy = MonitorImage("{} Image Test {} Noisy".format(args.val_dataset,
                                                                            args.noise_dist), 
                                            monitor,
                                            num_images=1, 
                                            normalize_method=normalize_method, 
                                            interval=1)
    monitor_image_test_recon = MonitorImage("{} Image Test {} Recon".format(args.val_dataset,
                                                                            args.noise_dist), 
                                            monitor,
                                            num_images=1, 
                                            normalize_method=normalize_method, 
                                            interval=1)

    # Evaluate
    for i in range(ds.size):
        # Read data
        x_data = di.next()[0]  # DI return as tupple
        
        # Create model
        x = nn.Variable([1, 3, x_data.shape[2], x_data.shape[3]])
        x.persistent = True
        x_noise = nn.Variable([1, 3, x_data.shape[2], x_data.shape[3]])
        x_noise.persistent = True
        x_recon = net(x_noise)
        x_recon.persistent = True
        x.d = x_data
        x_noise.d = apply_noise(x_data, args.noise_level, distribution=args.noise_dist, fix=True)

        # Forward (denoise)
        x_recon.forward(clear_buffer=True)

        # Log
        monitor_loss.add(i, psnr(x_recon.d, x_data))
        monitor_image_test_clean.add(i, x_data)
        monitor_image_test_noisy.add(i, x_noise.d)
        monitor_image_test_recon.add(i, x_recon.d)

        # Clear memory since the input is varaible size.
        import nnabla_ext.cuda
        nnabla_ext.cuda.clear_memory_cache()


def main():
    args = get_args()
    save_args(args)

    evaluate(args)


if __name__ == '__main__':
    main()

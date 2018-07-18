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

from datasets import data_iterator_imagenet
from args import get_args, save_args
from models import get_loss

def train(args):
    # Communicator and Context
    extension_module = args.context
    ctx = get_extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)

    # Model


    # Solver

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
    monitor_image_train_lr = MonitorImageTile("Image Tile Train LR",
                                              monitor,
                                              num_images=4, 
                                              normalize_method=normalize_method, 
                                              interval=1)
    monitor_image_train_hr = MonitorImageTile("Image Tile Train HR",
                                              monitor,
                                              num_images=4,
                                              normalize_method=normalize_method, 
                                              interval=1)
    # DataIterator
    rng = np.random.RandomState(410)
    di = data_iterator_imagenet(args.train_data_path, args.batch_size, rng=rng)
    
    # Train loop
    for i in range(args.max_iter):


        
        # Monitor
        monitor_loss.add(i, loss.d)
        monitor_time.add(i)

def main():
    args = get_args()
    save_args(args)

    train(args)

if __name__ == '__main__':
    main() 

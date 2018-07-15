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

from helpers import MonitorImageTileWithName
from models import generator, discriminator, gan_loss
from imagenet_data import data_iterator_imagenet

def train(args):
    # Communicator and Context
    extension_module = "cudnn"
    ctx = get_extension_context(extension_module)
    nn.set_default_context(ctx)

    # Model
    
    # Solver

    # Monitor
    monitor = Monitor(args.monitor_path)
    monitor_loss = MonitorSeries("Reconstruction Loss", monitor, interval=10)
    monitor_time = MonitorTimeElapsed("Training Time per Resolution", monitor, interval=10)
    monitor_image_tile_train = MonitorImageTileWithName("Image Tile Train", monitor,
                                                        num_images=args.batch_size)
    monitor_image_tile_test = MonitorImageTileWithName("Image Tile Test", monitor,
                                                        num_images=args.batch_size)

    # DataIterator
    rng = np.random.RandomState(410)

    
    # Train loop
    for i in range(args.max_iter):
        pass

def main():
    args = get_args()
    save_args(args)

    train(args)

if __name__ == '__main__':
    main() 

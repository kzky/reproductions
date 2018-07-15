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
    x = nn.Variable([args.batch_size, 3, args.ih, args.iw])
    x_noise = nn.Variable([args.batch_size, 3, args.ih, args.iw])
    

    # Data iterator
    di = data_iterator_kodak()


def main():
    args = get_args()
    save_args(args)

    evaluate(args)


if __name__ == '__main__':
    main()

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

from datasets import data_iterator_lapsrn
from args import get_args, save_args
from models import get_loss, lapsrn
from helpers import get_solver, downsample

import cv2


def train(args):
    # Context
    extension_module = args.context
    ctx = get_extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)

    # Model



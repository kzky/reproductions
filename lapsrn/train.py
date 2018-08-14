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
from functools import reduce

from datasets import data_iterator_lapsrn
from args import get_args, save_args
from models import get_loss, lapsrn
from helpers import (get_solver, upsample, downsample, 
                     split, to_BCHW, to_BHWC, normalize, ycbcr_to_rgb, 
                     normalize_method)

def train(args):
    # Context
    extension_module = args.context
    ctx = get_extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)

    # Model
    x_HR = nn.Variable([args.batch_size, 1, args.ih, args.iw])
    x_LRs = [x_HR]
    for s in range(args.S):
        x_LR = nn.Variable([args.batch_size, 1, args.ih // (2 ** (s+1)), args.iw // (2 ** (s+1))])
        x_LR.persistent = True
        x_LRs.append(x_LR)
    x_LRs = x_LRs[::-1]  # [x_LR0, x_LR1, ..., x_HR]
    x_SRs = lapsrn(x_LR, args.maps, args.S, args.R, args.D, args.skip_type, 
                   args.use_bn, share_type=args.share_type)
    for s in range(args.S):
        x_SRs[s].persistent = True
    loss = reduce(lambda x, y: x + y, 
                  [F.mean(get_loss(args.loss)(x, y)) for x, y in zip(x_LRs[1:], x_SRs)])

    # Solver
    solver = get_solver(args.solver)(args.lr)
    solver.set_parameters(nn.get_parameters())
    
    # Monitor
    monitor = Monitor(args.monitor_path)
    monitor_loss = MonitorSeries("Reconstruction Loss", monitor, interval=10)
    monitor_time = MonitorTimeElapsed("Training Time", monitor, interval=10)
    monitor_image_lr_list = []
    monitor_image_sr_list = []
    for s in range(args.S):
        monitor_image_lr = MonitorImageTile("Image Tile Train {} LR".format(2**(s+1)),
                                            monitor,
                                            num_images=4, 
                                            normalize_method=normalize_method, 
                                            interval=1)
        monitor_image_lr_list.append(monitor_image_lr)
        monitor_image_sr = MonitorImageTile("Image Tile Train {} SR".format(2**(s+1)),
                                            monitor,
                                            num_images=4, 
                                            normalize_method=normalize_method, 
                                            interval=1)
        monitor_image_sr_list.append(monitor_image_sr)

    # DataIterator
    from os.path import expanduser
    home = expanduser("~")
    img_paths = ["{}/nnabla_data/BSDS200".format(home), 
                 "{}/nnabla_data/General100".format(home), 
                 "{}/nnabla_data/T91".format(home)]
    di = data_iterator_lapsrn(img_paths, batch_size=args.batch_size, shuffle=True)
    
    # Train loop
    for i in range(args.max_iter):
        # Feed data
        x_data = di.next()[0]
        ycbcrs = []
        x_LR_d = x_data  # B, H, W, C
        for s, x_LR in enumerate(x_LRs[::-1]):  # [x_HR, ..., x_LR1, x_LR0]
            x_LR_y, x_LR_cb, x_LR_cr = split(x_LR_d)           
            ycbcrs.append([x_LR_y, x_LR_cb, x_LR_cr])
            x_LR_y = to_BCHW(x_LR_y)
            x_LR_y = normalize(x_LR_y)
            x_LR.d = x_LR_y
            x_LR_d = downsample(x_data, 2 ** (s + 1))
        ycbcr = ycbcrs[-1]

        # Zerograd, forward, backward, weight-decay, update
        solver.zero_grad()
        loss.forward(clear_no_need_grad=True)
        loss.backward(clear_buffer=True)
        solver.weight_decay(args.decay_rate)
        solver.update()

        # LR decay
        if i in args.decay_at:
            solver.set_learning_rate(solver.learning_rate() * 0.1)
        
        # Monitor and save
        monitor_loss.add(i, loss.d)
        monitor_time.add(i)
        if i % args.save_interval == 0:
            for s in range(args.S):
                _, cb, cr = ycbcr
                cb = upsample(cb, 2 ** (s + 1))
                cr = upsample(cr, 2 ** (s + 1))
                x_lr = to_BCHW(ycbcr_to_rgb(to_BHWC(x_LRs[s+1].d) * 255.0, cb, cr))
                x_sr = to_BCHW(ycbcr_to_rgb(to_BHWC(np.clip(x_SRs[s].d, 0.0, 1.0) * 255.0), cb, cr))
                monitor_image_lr_list[s].add(i, x_lr)
                monitor_image_sr_list[s].add(i, x_sr)
            nn.save_parameters("{}/param_{}.h5".format(args.monitor_path, i))

    # Monitor and save
    monitor_loss.add(i, loss.d)
    monitor_time.add(i)
    for s in range(args.S):
        _, cb, cr = ycbcr
        cb = upsample(cb, 2 ** (s + 1))
        cr = upsample(cr, 2 ** (s + 1))
        x_lr = to_BCHW(ycbcr_to_rgb(to_BHWC(x_LRs[s+1].d) * 255.0, cb, cr))
        x_sr = to_BCHW(ycbcr_to_rgb(to_BHWC(x_SRs[s].d) * 255.0, cb, cr))
        monitor_image_lr_list[s].add(i, x_lr)
        monitor_image_sr_list[s].add(i, x_sr)
    nn.save_parameters("{}/param_{}.h5".format(args.monitor_path, i))


def main():
    args = get_args()
    save_args(args, "train")

    train(args)

if __name__ == '__main__':
    main() 

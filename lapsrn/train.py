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
from helpers import get_solver

import cv2

def train(args):
    # Context
    extension_module = args.context
    ctx = get_extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)

    # Model
    x_HR = nn.Variable([args.batch_size, 3, args.ih, args.iw])
    x_LRs = [x_HR]
    for s in range(args.S):
        x_LR = nn.Variable([args.batch_size, 3, args.ih // (2 ** (s+1)), args.iw // (2 ** (s+1))])
        x_LRs.append(x_LR)
    x_LRs = x_LRs[::-1]
    x_SRs = lapsrn(x_LR, args.maps, args.S, args.R, args.D, args.skip_type, args.use_bn)
    loss = reduce(lambda x, y: x + y, 
                  [F.mean(get_loss(args.loss)(x, y)) for x, y in zip(x_LRs[1:], x_SRs)])

    # Solver
    solver = get_solver(args.solver)(args.lr)
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
    monitor_time = MonitorTimeElapsed("Training Time", monitor, interval=10)
    monitor_image_lr_list = []
    for s in range(args.S):
        monitor_image_lr = MonitorImageTile("Image Tile Train LR",
                                            monitor,
                                            num_images=4, 
                                            normalize_method=normalize_method, 
                                            interval=1)
        monitor_image_lr_list.append(monitor_image_lr)
    monitor_image_hr = MonitorImageTile("Image Tile Train HR",
                                        monitor,
                                        num_images=4,
                                        normalize_method=normalize_method, 
                                        interval=1)
    # DataIterator
    img_paths = ["/home/kzky/nnabla_data/BSD200", 
                 "/home/kzky/nnabla_data/General100", 
                 "/home/kzky/nnabla_data/T90"]
    #TODO: should normalize?
    di = data_iterator_lapsrn(img_paths, batch_size=args.batch_size, shuffle=True)
    
    # Train loop
    for i in range(args.max_epoch):
        x_data = di.next()[0]
        x_HR.d = x_data
        x_data_s = x_data
        for x_LR in x_LRs[:-1][::-1]:
            b, c, h, w = x_data_s.shape
            x_data_s = x_data_s.transpose(0, 2, 3, 1)
            x_data_s = np.asarray([cv2.resize(x, (h // 2, w // 2), interpolation=cv2.INTER_CUBIC) \
                                   for x in x_data_s])
            x_data_s = x_data_s.transpose(0, 3, 1, 2)
            x_LR.d = x_data_s
            
        solver.zero_grad()
        loss.forward(clear_no_need_grad=True)
        loss.backward(clear_buffer=True)
        solver.weight_decay(args.decay_rate)
        solver.update()

        if i in args.decay_at:
            solver.set_learning_rate(solver.learning_rate() * 0.5)
        
        # Monitor and save
        monitor_loss.add(i, loss.d)
        monitor_time.add(i)
        if i % args.save_interval == 0:
            for s in range(args.S):
                monitor_image_lr_list[s].add(i, x_LRs[s].d.copy())
            monitor_image_hr.add(i, x_HR.d.copy())
            nn.save_parameters("{}/param_{}.h5".format(args.monitor_path, i))
            

def main():
    args = get_args()
    save_args(args, "train")

    train(args)

if __name__ == '__main__':
    main() 

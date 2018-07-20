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
    # Context
    extension_module = args.context
    ctx = get_extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)

    # Model
    x_HR = nn.Variable([args.batch_size, 3, args.ih, args.iw])
    x_LRs = [x_HR]
    for s in range(args.S):
        x_LR = F.average_pooling(x_LRs[-1], (2**s, 2**s))
        x_LRs.append(x_LR)
    x_LRs = x_LRs[:0:-1]
    x_SRs = lapsrn(x_LR, args.maps, args.S, args.R, args.D, args.skip_type)
    loss = reduce(lambda x, y: x + y, 
                  [F.mean(get_loss(args.loss)(x, y)) for x, y in zip(x_LRs, x_SRs)])

    # Solver
    solver = S.Momentum(args.lr)
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
    #TODO
    # augmentation:
    #    - random downsample: [0.5, 1.0]
    #    - rotation: 90, 180, 270
    #    - flip: horizontal
    
    # DataIterator
    rng = np.random.RandomState(410)
    di = data_iterator_imagenet(args.train_data_path, args.batch_size, rng=rng)
    
    # Train loop
    for i in range(args.max_epoch):
        x_HR.d = di.next()
        solver.zero_grad()
        loss.forward(clear_no_need_grad=True)
        solver.backward(clear_buffer=True)

        if i in args.decay_at:
            solver.set_learning_rate(solver.learning_rate() * 0.5)
        
        # Monitor
        monitor_loss.add(i, loss.d)
        monitor_time.add(i)
        if i % 1000 == 0:
            for s in range(args.S):
                monitor_image_lr_list[s].add(x_LRs[s].d.copy())
            monitor_image_hr.add(x_HR.d.copy())

def main():
    args = get_args()
    save_args(args)

    train(args)

if __name__ == '__main__':
    main() 

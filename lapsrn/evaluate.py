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

import cv2

from helpers import psnr, downsample
from args import get_args, save_args
from models import lapsrn
from datasets import data_iterator_lapsrn


def resize(x, s):
    b, c, h, w = x.shape
    x = x.reshape(c, h, w)
    x = x.transpose(1, 2, 0)
    x = cv2.resize(x, (h * 2 ** s, w * 2 ** s), interpolation=cv2.INTER_CUBIC)
    x = x.transpose(2, 0, 1)
    c, h, w = x.shape
    x = x.reshape(b, c, h, w)

    return x
    

def evaluate(args):
    # Context
    ctx = get_extension_context(args.context, device_id=args.device_id)
    nn.set_default_context(ctx)

    # Model
    nn.load_parameters(args.model_load_path)

    # Data iterator
    img_paths = args.valid_data_path
    di = data_iterator_lapsrn(img_paths, batch_size=1, train=False, shuffle=False)

    # Monitor
    def normalize_method(x):
        max_idx = np.where(x > 255)
        min_idx = np.where(x < 0)
        x[max_idx] = 255
        x[min_idx] = 0
        return x
    monitor = Monitor(args.monitor_path)
    valid_data_name = args.valid_data_path.rstrip("/").split("/")[-1]
    monitor_metric = MonitorSeries("Evaluation Metric {} {}".format(args.valid_metric, 
                                                                    valid_data_name), 
                                   monitor, interval=1)
    monitor_image_lr = MonitorImage("Image Test LR {}".format(valid_data_name),
                                    monitor,
                                    num_images=1, 
                                    normalize_method=normalize_method, 
                                    interval=1)
    monitor_image_hr = MonitorImage("Image Test HR {}".format(valid_data_name),
                                    monitor,
                                    num_images=1, 
                                    normalize_method=normalize_method, 
                                    interval=1)
    monitor_image_sr_list = []
    for s in range(args.S):
        monitor_image_sr = MonitorImage("Image Test SR x{} {}".format(2 **(s + 1), valid_data_name),
                                        monitor,
                                        num_images=1, 
                                        normalize_method=normalize_method, 
                                        interval=1)
        monitor_image_sr_list.append(monitor_image_sr)

    # Evaluate
    for i in range(di.size):
        # Read data
        x_data = di.next()[0]  # DI return as tupple

        # Create model
        x_HR = nn.Variable([1, 3, x_data.shape[2], x_data.shape[3]])
        x_LRs = [x_HR]
        x_LRs = [x_HR]
        _, _, ih, iw = x_data.shape
        for s in range(args.S):
            x_LR = nn.Variable([1, 3, ih // (2 ** (s+1)), iw // (2 ** (s+1))])
            x_LRs.append(x_LR)
        x_LRs = x_LRs[::-1]
        x_SRs = lapsrn(x_LR, args.maps, args.S, args.R, args.D, args.skip_type, 
                       args.use_bn, test=False)
        for x_SR in x_SRs:
            x_SR.persistent = True
        x_LR.persistent = True
        x_SR = x_SRs[-1]

        # Feed data
        x_HR.d = x_data
        x_data_s = x_data
        for x_LR in x_LRs[:-1][::-1]:
            x_data_s = downsample(x_data_s)
            x_LR.d = x_data_s

        # Forward
        x_SR.forward(clear_buffer=True)
        
        # Log
        try:
            #todo: how to?
            monitor_metric.add(i, psnr(x_HR.d, x_SR.d))
        except:
            nn.logger.warn("{}-th image could not down-sampled well".format(i))
        monitor_image_lr.add(i, resize(x_LR.d.copy(), args.S))
        monitor_image_hr.add(i, x_HR.d.copy())
        for k, x_SR in enumerate(x_SRs):
            monitor_image_sr_list[k].add(i, x_SR.d.copy())

        # Clear memory since the input is varaible size.
        import nnabla_ext.cuda
        nnabla_ext.cuda.clear_memory_cache()


def main():
    args = get_args()
    save_args(args, "eval")

    evaluate(args)


if __name__ == '__main__':
    main()

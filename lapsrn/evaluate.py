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

from helpers import (get_solver, upsample, downsample, 
                     split, to_BCHW, to_BHWC, normalize, ycrcb_to_rgb, 
                     normalize_method)
from args import get_args, save_args
from models import lapsrn
from datasets import data_iterator_lapsrn


def resize(x, s):
    b, c, h, w = x.shape
    x = x.reshape(c, h, w)
    x = x.transpose(1, 2, 0)
    x = cv2.resize(x, (w * 2 ** s, h * 2 ** s), interpolation=cv2.INTER_CUBIC)
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
        x_data = di.next()[0]
        b, h, w, c = x_data.shape

        # Create model
        x_HR = nn.Variable([1, 1, h, w])
        x_LRs = [x_HR]
        for s in range(args.S):
            x_LR = nn.Variable([1, 1, h // (2 ** (s+1)), w // (2 ** (s+1))])
            x_LR.persistent = True
            x_LRs.append(x_LR)
        x_LRs = x_LRs[::-1]  # [x_LR0, x_LR1, ..., x_HR]
        x_SRs = lapsrn(x_LR, args.maps, args.S, args.R, args.D, args.skip_type, 
                       args.use_bn, share_type=args.share_type)
        for s in range(args.S):
            x_SRs[s].persistent = True
        x_SR = x_SRs[-1]

        # Feed data
        ycrcb = []
        x_LR_d = x_data  # B, H, W, C
        for s, x_LR in enumerate(x_LRs[::-1]):  # [x_HR, ..., x_LR1, x_LR0]
            x_LR_y, x_LR_cr, x_LR_cb = split(x_LR_d)            
            ycrcb.append([x_LR_y, x_LR_cr, x_LR_cb])
            x_LR_y = to_BCHW(x_LR_y)
            x_LR_y = normalize(x_LR_y)
            x_LR.d = x_LR_y
            x_LR_d = downsample(x_data, 2 ** (s + 1))

        ycrcb = ycrcb[-1]

        # Forward
        x_SR.forward(clear_buffer=True)
        
        # Log
        try:
            #todo: how to?
            #todo: only y of YCrCB?
            monitor_metric.add(i, psnr(x_HR.d, x_SR.d))
        except:
            nn.logger.warn("{}-th image could not down-sampled well".format(i))
        
        x_lr = to_BHWC(x_LR.d.copy()) * 255
        x_lr = x_lr.astype(np.uint8)
        x_lr = upsample(x_lr, 2 ** args.S)
        x_hr = to_BHWC(x_HR.d.copy() * 255)
        _, cr, cb = ycrcb
        cr = upsample(cr, 2 ** args.S)
        cb = upsample(cb, 2 ** args.S)
        monitor_image_lr.add(i, to_BCHW(ycrcb_to_rgb(x_lr, cr, cb)))
        monitor_image_hr.add(i, to_BCHW(ycrcb_to_rgb(x_hr, cr, cb)))
        for s, x_SR in enumerate(x_SRs):
            _, cr, cb = ycrcb
            cr = upsample(cr, 2 ** (s + 1))
            cb = upsample(cb, 2 ** (s + 1))
            x_sr = to_BCHW(ycrcb_to_rgb(to_BHWC(x_SRs[s].d.copy()) * 255.0, cr, cb))
            monitor_image_sr_list[s].add(i, x_sr)

        # Clear memory since the input is varaible size.
        import nnabla_ext.cuda
        nnabla_ext.cuda.clear_memory_cache()


def main():
    args = get_args()
    save_args(args, "eval")

    evaluate(args)


if __name__ == '__main__':
    main()

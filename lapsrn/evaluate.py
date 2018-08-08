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

from helpers import (get_solver, resize, upsample, downsample,
                     split, to_BCHW, to_BHWC, normalize, ycrcb_to_rgb, 
                     normalize_method, denormalize,
                     psnr)
from args import get_args, save_args
from models import lapsrn
from datasets import data_iterator_lapsrn


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
    monitor_psnr_sr_list = []
    monitor_psnr_lr = MonitorSeries("Evaluation Metric PSNR Bicubic-{} {}".format(2 ** args.S, valid_data_name), 
                                    monitor, interval=1)
    for s in range(args.S):
        monitor_psnr_sr = MonitorSeries("Evaluation Metric PSNR SR-{} {}".format(2 ** (s + 1), valid_data_name), 
                                        monitor, interval=1)
        monitor_psnr_sr_list.append(monitor_psnr_sr)
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
        if h % (2 ** args.S) != 0 or w % (2 ** args.S) != 0:
            h_ = h + 2 ** args.S - h % (2 ** args.S)
            w_ = w + 2 ** args.S - w % (2 ** args.S)
            x_data = resize(x_data, h_, w_)
            b, h, w, c = x_data.shape
            nn.logger.warn("Input shape is coerced to (h, w, c)=({})".format(h, w, c))

        # Create model
        x_HR = nn.Variable([1, 1, h, w])
        x_LRs = [x_HR]
        for s in range(args.S):
            sh = h // (2 ** (s+1))
            sw = w // (2 ** (s+1))
            x_LR = nn.Variable([1, 1, sh, sw])
            x_LR.persistent = True
            x_LRs.append(x_LR)
        x_LRs = x_LRs[::-1]  # [x_LR0, x_LR1, ..., x_HR]
        x_SRs = lapsrn(x_LR, args.maps, args.S, args.R, args.D, args.skip_type, 
                       args.use_bn, test=True, share_type=args.share_type)
        for s in range(args.S):
            x_SRs[s].persistent = True
        x_SR = x_SRs[-1]

        # Feed data
        ycrcbs = []
        x_LR_d = x_data  # B, H, W, C
        for s, x_LR in enumerate(x_LRs[::-1]):  # [x_HR, ..., x_LR1, x_LR0]
            x_LR_y, x_LR_cr, x_LR_cb = split(x_LR_d)            
            ycrcbs.append([x_LR_y, x_LR_cr, x_LR_cb])
            x_LR_y = to_BCHW(x_LR_y)
            x_LR_y = normalize(x_LR_y)
            x_LR.d = x_LR_y
            x_LR_d = downsample(x_data, 2 ** (s + 1))
        ycrcb = ycrcbs[-2]

        # Forward
        x_SR.forward(clear_buffer=True)

        _, cr, cb = ycrcb
        cr = upsample(cr, 2 ** (args.S-1))
        cb = upsample(cb, 2 ** (args.S-1))
        x_lr = denormalize(to_BHWC(x_LR.d))
        x_lr = x_lr.astype(np.uint8)
        x_lr = upsample(x_lr, 2 ** args.S)
        x_lr = to_BCHW(ycrcb_to_rgb(x_lr, cr, cb))

        x_hr = denormalize(to_BHWC(x_HR.d))
        cr, cb = ycrcbs[0][1], ycrcbs[0][1]
        x_hr = to_BCHW(ycrcb_to_rgb(x_hr, cr, cb))
        monitor_image_lr.add(i, x_lr)
        monitor_image_hr.add(i, x_hr)
        monitor_psnr_lr.add(i, psnr(x_hr, x_lr))
        for s, x_SR in enumerate(x_SRs):
            _, cr, cb = ycrcb
            cr = upsample(cr, 2 ** s)
            cb = upsample(cb, 2 ** s)
            x_sr = to_BCHW(ycrcb_to_rgb(denormalize(to_BHWC(np.clip(x_SRs[s].d, 0.0, 1.0))), cr, cb))
            monitor_image_sr_list[s].add(i, x_sr)

        monitor_psnr_sr.add(i, psnr(x_hr, x_sr))

        # Clear memory since the input is varaible size.
        import nnabla_ext.cuda
        nnabla_ext.cuda.clear_memory_cache()


def main():
    args = get_args()
    save_args(args, "eval")

    evaluate(args)


if __name__ == '__main__':
    main()

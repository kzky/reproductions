import os
import numpy as np
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save
from nnabla.ext_utils import get_extension_context

from imagenet_data import data_iterator_imagenet
from models import generator
from models import is_model


def compute_inception_score(di, args):
    preds = []
    ys = []
    for i in range(itr):
        # Read
        x_data, y_data = di.next()
        # Generate
        z = F.randn(0, 1.0, [args.batch_size, args.latent])
        x_fake = generator(z, y, maps=args.latent, sn=args.not_sn)
        # Predict
        pred = F.softmax(is_model(x_fake))
        preds.append(pred.d)
        y.append(y_data)
    # Score
    y_approx = np.mean(ys, axis=(0, 1))
    score = 0
    for pred in preds:
        score += np.sum(pred * np.log(pred / y_approx))
    return score
        

def compute_frechet_inception_distance(di, args):
    h_fakes = []
    h_reals = []
    for i in range(itr):
        # Read
        x_data, y_data = di.next()
        # Generate
        z = F.randn(0, 1.0, [args.batch_size, args.latent])
        x_real = nn.Variable.from_numpy_array(x_data)
        x_fake = generator(z, y, maps=args.latent, sn=args.not_sn)
        # Predict
        h_fake = is_model_coding_layer(x_fake)
        h_real = is_model_coding_layer(x_real)
        for hf, hr in zip(h_fake.d, h_real.d):
            h_fakes.append(hf)
            h_reals.append(hr)
    h_fakes = np.asarray(h_fakes)
    h_reals = np.asarray(h_reals)

    # FID score
    ave_h_real = np.mean(h_reals, axis=0)    
    ave_h_fake = np.mean(h_fakes, axis=0)
    cov_h_fake = np.dot(h_fakes.T, h_fakes) / len(f_fakes) - np.dot(ave_h_fake.T, ave_h_fake)
    cov_h_real = np.dot(h_reals.T, h_reals) / len(f_reals) - np.dot(ave_h_real.T, ave_h_real)
    score = np.sum((ave_h_real - ave_h_fake)**2) \
            + np.trace(cov_h_real + cov_h_fake - 2 * np.sqrt(np.dot(cov_h_real.T, cov_h_fake)))
    return score

def validate(args):
    # Context
    ctx = get_extension_context(args.context, device_id=args.device_id)
    nn.set_default_context(ctx)
    nn.set_auto_forward(True)

    # Generator
    nn.load_parameters(args.model_load_path)
    
    # Inception model
    nn.load_parameters(args.inception_model_load_path)

    if args.validation_metric == "IS":
        is_model = None
        compute_metric = compute_inception_score_batch
    if args.validation_metric == "FID":
        is_model = None
        compute_metric = compute_frechet_inception_distance

    # Data Iterator
    di = data_iterator_imagenet(args.batch_size, args.val_cachefile_dir)

    # Compute the validation metric
    score = compute_metric(x_fake, is_model, di, itr)

    # Monitor
    monitor = Monitor(args.monitor_path)
    monitor_metric = MonitorSeries("{}".format(args.validation_metric), monitor, interval=1)
    monitor_metric.add(score)


def main():
    args = get_args()
    save_args(args)

    validate(args)
    

if __name__ == '__main__':
    main()

                        

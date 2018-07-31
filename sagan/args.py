# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def get_args(batch_size=16, accum_grad=4, image_size=128, n_classes=1000, max_iter=200000):
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    import argparse
    import os

    description = "Example of Self-Attention GAN (SAGAN)."
    parser = argparse.ArgumentParser(description)

    parser.add_argument("-d", "--device-id", type=str, default="0",
                        help="Device id.")
    parser.add_argument("-c", "--context", type=str, default="cudnn",
                        help="Context.")
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument("--image-size", type=int, default=image_size,
                        help="Image size.")
    parser.add_argument("--n-classes", type=int, default=n_classes,
                        help="Image size.")
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size,
                        help="Batch size.")
    parser.add_argument("--accum-grad", "-a", type=int, default=accum_grad,
                        help="Batch size.")
    parser.add_argument("--max-iter", "-i", type=int, default=max_iter,
                        help="Max iterations.")
    parser.add_argument("--save-interval", type=int, required=True,
                        help="Interval for saving models.")
    parser.add_argument("--latent", type=int, default=128,
                        help="Number of latent variables.")
    parser.add_argument("--maps", type=int, default=1024,
                        help="Number of latent variables.")
    parser.add_argument("--not-sn", action='store_false',
                        help="Not use the spectral normalization")
    parser.add_argument("--monitor-path", type=str, default="./result/example_0",
                        help="Monitor path.")
    parser.add_argument("--model-load-path", type=str,
                        help="Model load path to a h5 file used in generation and validation.")
    parser.add_argument("--inception-model-load-path", type=str,
                        help="Inception model load path to a h5 file used in validation.")
    parser.add_argument("--lrg", type=float, default=1 * 1e-4,
                        help="Learning rate for generator")
    parser.add_argument("--lrd", type=float, default=4 * 1e-4,
                        help="Learning rate for discriminator")
    parser.add_argument("--beta1", type=float, default=0.0,
                        help="Beta1 of Adam solver.")
    parser.add_argument("--beta2", type=float, default=0.9,
                        help="Beta2 of Adam solver.")
    parser.add_argument("--train-cachefile-dir", "-T", type=str, default="",
                        help='Training cache file dir. Create to use create_cache_file.py')
    parser.add_argument("--val-cachefile-dir", "-V", type=str, default="",
                        help='Validation cache file dir. Create to use create_cache_file.py')
    parser.add_argument("--hyper-sphere", action='store_true',
                        help="Latent vector lie in the hyper sphere.")
    parser.add_argument("--validation-metric", type=str, default="FID",
                        choices=["IS", "FID"],
                        help="Validation metric for SAGAN; IS or FID, FID is the default.")
    parser.add_argument("--sync-weight-every-itr",
                        type=int, default=10,
                        help="Sync weights every specified iteration. NCCL uses the ring all reduce, so gradients in each device are not exactly same. When it is accumulated in the weights, the weight values in each device diverge.")

    args = parser.parse_args()

    return args


def save_args(args):
    from nnabla import logger
    import os
    if not os.path.exists(args.monitor_path):
        os.makedirs(args.monitor_path)

    path = "{}/Arguments.txt".format(args.monitor_path)
    logger.info("Arguments are saved to {}.".format(path))
    with open(path, "w") as fp:
        for k, v in sorted(vars(args).items()):
            logger.info("{}={}".format(k, v))
            fp.write("{}={}\n".format(k, v))

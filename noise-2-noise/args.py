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


def get_args(batch_size=32, ih=256, iw=256, max_iter=200000):
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
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size,
                        help="Batch size.")
    parser.add_argument("--ih", type=int, default=ih,
                        help="Image height.")
    parser.add_argument("--iw", type=int, default=iw,
                        help="Image width.")
    parser.add_argument("--max-iter", "-i", type=int, default=max_iter,
                        help="Max iterations.")
    parser.add_argument("--net", type=int, default="RED",
                        choices=["RED", "noise2noise"]
                        help="RED or noise2noise")
    parser.add_argument("--loss", type=int, default="l2",
                        choices=["l2", "l1"]
                        help="Loss")
    parser.add_argument("--dist", type=int, default="gaussian",
                        help="Noise distribution", 
                        choices=["gaussian", "poisson", "bernoulli"])
    parser.add_argument("--noise-level", type=int, default=50,
                        help="Noise level")
    parser.add_argument("--save-interval", type=int, required=True,
                        help="Interval for saving models.")
    parser.add_argument("--monitor-path", type=str, default="./result/example_0",
                        help="Monitor path.")
    parser.add_argument("--model-load-path", type=str,
                        help="Model load path to a h5 file used in generation and validation.")
    parser.add_argument("--lr", type=float, default=1 * 1e-3,
                        help="Learning rate for generator")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Beta1 of Adam solver.")
    parser.add_argument("--beta2", type=float, default=0.99,
                        help="Beta2 of Adam solver.")
    parser.add_argument("--train-data-path", "-T", type=str, default="",
                        help='Path to training data')
    parser.add_argument("--val-data-path", "-V", type=str, default="",
                        help='Path to validation data')
    parser.add_argument("--validation-metric", type=str, default="",
                        choices=[],
                        help="Validation metric for Noise2Noise.")
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

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


def get_args(batch_size=16, ih=128, iw=128, max_iter=150 * 1000, save_interval=10 * 1000, 
             S=3, R=8, D=5, maps=64):
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    import argparse
    import os

    description = "Example of Noise-2-Noise netowrk."
    parser = argparse.ArgumentParser(description)

    parser.add_argument("-d", "--device-id", type=str, default="0",
                        help="Device id.")
    parser.add_argument("-c", "--context", type=str, default="cudnn",
                        help="Context.")
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size,
                        help="Batch size.")
    parser.add_argument("--train", type=bool, default=True,
                        help="Train mode")
    parser.add_argument("--ih", type=int, default=ih,
                        help="Image height.")
    parser.add_argument("--iw", type=int, default=iw,
                        help="Image width.")
    parser.add_argument("--max-iter", type=int, default=max_iter,
                        help="Max iterations")
    parser.add_argument("--S", type=int, default=S,
                        help="Super resolution number.")
    parser.add_argument("--R", type=int, default=R,
                        help="Recursive number.")
    parser.add_argument("--D", type=int, default=D,
                        help="Depth for recursive block.")
    parser.add_argument("--maps", type=int, default=maps,
                        help="Number of maps")
    parser.add_argument("--skip-type", type=str, default="ss",
                        choices=["ss", "ds", "ns"], 
                        help="Skip type")
    parser.add_argument("--loss", type=str, default="charbonnier",
                        choices=["charbonnier", "l2", "l1"],
                        help="Loss")
    parser.add_argument("--use-bn", action='store_true',
                        help='Use batch norm')
    parser.add_argument("--share-type", default="across-pyramid", 
                        choices=["across-pyramid", "within-pyramid"], 
                        help='Share type')
    parser.add_argument("--save-interval", type=int, default=save_interval,
                        help="Interval for saving models.")
    parser.add_argument("--monitor-path", type=str, default="./result/example_0",
                        help="Monitor path.")
    parser.add_argument("--model-load-path", type=str,
                        help="Model load path to a h5 file used in generation and validation.")
    parser.add_argument("--solver", type=str, default="Momentum", 
                        help="Solver")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate for generator")
    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="Learning rate for generator")
    parser.add_argument("--img-paths", type=str, nargs='+', 
                        default=["BSDS200", 
                                 "General100", 
                                 "T91"],
                        help="List of path to dataset")
    parser.add_argument("--decay-at", "-D", type=int, nargs='+',
                        default=[50000, 100000, 150000],
                        help="Decay-at `iteration` for learning rate.")
    parser.add_argument("--lr-decay-rate", type=float, default=0.1,  
                        help="Learning rate decay rate")
    parser.add_argument("--weight-decay-rate", type=float, default=1e-4, 
                        help="Weight decay rate")
    parser.add_argument("--train-data-path", "-T", type=str, default="",
                        help='Path to training data')
    parser.add_argument("--valid-data-path", "-V", type=str, default="",
                        help='Validation data to be used')
    parser.add_argument("--imresize-mode", type=str, default="opencv",
                        help='Image resize mode. Default is `opnecv`.')
    #TODO: add SSIM, IFC
    parser.add_argument("--valid-metric", type=str, default="psnr",
                        choices=["psnr"],
                        help="Validation metric for reconstruction.")
    args = parser.parse_args()

    return args


def save_args(args, mode="train"):
    from nnabla import logger
    import os
    if not os.path.exists(args.monitor_path):
        os.makedirs(args.monitor_path)

    path = "{}/Arguments-{}.txt".format(args.monitor_path, mode)
    logger.info("Arguments are saved to {}.".format(path))
    with open(path, "w") as fp:
        for k, v in sorted(vars(args).items()):
            logger.info("{}={}".format(k, v))
            fp.write("{}={}\n".format(k, v))

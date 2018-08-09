#TODO: set better default hyper parameters


def get_args(monitor_path='tmp.monitor', max_epoch=50, 
             learning_rate=2*1e-4,
             batch_size=32):
    import argparse
    import os
    description = ("NNabla implementation of AugmentedCycleGAN")
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size)
    parser.add_argument("--learning-rate", "-l",
                        type=float, default=learning_rate)
    parser.add_argument("--monitor-path", "-m",
                        type=str, default=monitor_path,
                        help='Path monitoring logs saved.')
    parser.add_argument("--max-epoch", "-e", type=int, default=max_epoch,
                        help='Max epoch of training. Epoch is determined by the max of the number of images for two domains.')
    parser.add_argument("--decay-at", type=int, default=max_epoch / 2,
                        help='Linear learning rate starts from decay-at `epoch`.')
    parser.add_argument("--device-id", "-d", type=int, default=0,
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--model-load-path",
                        type=str,
                        help='Path where model parameters are loaded.')
    parser.add_argument("--dataset",
                        type=str, default="edges2shoes", choices=["cityscapes",
                                                                  "edges2handbags",
                                                                  "edges2shoes",
                                                                  "facades",
                                                                  "maps"],
                        help='Dataset to be used.')
    parser.add_argument('--context', '-c', type=str,
                        default=None, help="Extension modules. ex) 'cpu', 'cudnn'.")
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type configuration (float or half)')
    parser.add_argument('--lam', type=float, default=0.025,
                        help="Coefficient for reconstruction loss.")
    parser.add_argument('--latent', default=16, 
                        help="Number of dimensions of latent variables.")
    parser.add_argument('--unpool', action="store_true",
                        help="Number of dimensions of latent variables.")
    parser.add_argument('--num-data', default=-1, type=int, 
                        help="Number of training data to be used. Default is 1, using all data.")
    
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

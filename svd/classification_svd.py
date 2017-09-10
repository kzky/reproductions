from __future__ import absolute_import
from six.moves import range

import os

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save

from args import get_args
from mnist_data import data_iterator_mnist
from models import mnist_lenet_prediction, categorical_error, kl_divergence

def classification_svd():
    args = get_args()

    # Get context.
    from nnabla.contrib.context import extension_context
    extension_module = args.context
    if args.context is None:
        extension_module = 'cpu'
    logger.info("Running in %s" % extension_module)
    ctx = extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)

    # Create CNN network for both training and testing.
    mnist_cnn_prediction = mnist_lenet_prediction

    # TRAIN
    reference = "reference"
    slim = "slim"
    # Create input variables.
    image = nn.Variable([args.batch_size, 1, 28, 28])
    label = nn.Variable([args.batch_size, 1])
    # Create `reference` and "slim" prediction graph.
    model_load_path = args.model_load_path
    pred = mnist_cnn_prediction(image, net=slim, test=False)
    pred.persistent = True
    
    # SVD
    rrate = 0.5  # reduction rate
    decompose_network(model_load_path, reference, slim, rrate)
    loss = F.mean(F.softmax_cross_entropy(pred, label))

    # TEST
    # Create input variables.
    vimage = nn.Variable([args.batch_size, 1, 28, 28])
    vlabel = nn.Variable([args.batch_size, 1])
    # Create reference predition graph.
    vpred = mnist_cnn_prediction(vimage, net=slim, test=True)

    # Create Solver.
    solver = S.Adam(args.learning_rate)
    with nn.parameter_scope(slim):
        solver.set_parameters(nn.get_parameters())
    
    # Create monitor.
    from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
    monitor = Monitor(args.monitor_path)
    monitor_loss = MonitorSeries("Training loss", monitor, interval=10)
    monitor_err = MonitorSeries("Training error", monitor, interval=10)
    monitor_time = MonitorTimeElapsed("Training time", monitor, interval=100)
    monitor_verr = MonitorSeries("Test error", monitor, interval=10)

    # Initialize DataIterator for MNIST.
    data = data_iterator_mnist(args.batch_size, True)
    vdata = data_iterator_mnist(args.batch_size, False)
    best_ve = 1.0
    # Training loop.
    for i in range(args.max_iter):
        if i % args.val_interval == 0:
            # Validation
            ve = 0.0
            for j in range(args.val_iter):
                vimage.d, vlabel.d = vdata.next()
                vpred.forward(clear_buffer=True)
                ve += categorical_error(vpred.d, vlabel.d)
            monitor_verr.add(i, ve / args.val_iter)
        if ve < best_ve:
            nn.save_parameters(os.path.join(
                args.model_save_path, 'params_%06d.h5' % i))
            best_ve = ve
        # Training forward
        image.d, label.d = data.next()
        solver.zero_grad()
        loss.forward(clear_no_need_grad=True)
        loss.backward(clear_buffer=True)
        solver.weight_decay(args.weight_decay)
        solver.update()
        e = categorical_error(pred.d, label.d)
        monitor_loss.add(i, loss.d.copy())
        monitor_err.add(i, e)
        monitor_time.add(i)

    ve = 0.0
    for j in range(args.val_iter):
        vimage.d, vlabel.d = vdata.next()
        vpred.forward(clear_buffer=True)
        ve += categorical_error(vpred.d, vlabel.d)
    monitor_verr.add(i, ve / args.val_iter)

    parameter_file = os.path.join(
        args.model_save_path, 'params_{:06}.h5'.format(args.max_iter))
    nn.save_parameters(parameter_file)

if __name__ == '__main__':
    classification_svd()

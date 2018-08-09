import os
import numpy as np
import argparse
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.monitor import Monitor, MonitorImage, MonitorSeries, MonitorTimeElapsed
from nnabla.ext_utils import get_extension_context

from args import get_args, save_args
from datasets import pix2pix_data_source, pix2pix_data_iterator
from models import (G_xy, G_yx, D_x, D_y, E, D_z_x,
                    recon_loss, lsgan_loss, 
                    gauss_reparameterize, log_prob_gaussian, log_prob_laplace, log_prob_gaussian)


def linear_decay(solver, base_lr, epoch, decay_at, max_epoch):
    """Decay towards zero after `decay-at` epoch"""
    numerator = max(0, epoch - decay_at)
    lr = base_lr * (1. - numerator / float(max_epoch - numerator))
    logger.info("Msg:Learning rate decayed,epoch:{},lr:{}".format(epoch, lr))
    solver.set_learning_rate(lr)


def gradient_clipping(params, max_norm, norm_type=2):
    params = list(filter(lambda p: p.need_grad == True, params.values()))
    norm_type = float(norm_type)

    if norm_type == float('inf'):
        total_norm = max(np.abs(p.g).max() for p in params)
    else:
        total_norm = 0.
        for p in params:
            param_norm = F.pow_scalar(
                F.sum(p.grad ** norm_type), 1. / norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coeff = max_norm / (float(total_norm.data) + 1e-6)
    if clip_coeff < 1:
        for p in params:
            p.g = p.g * clip_coeff

def update_paramters(solver, params):
    #TODO: now imbalance
    if solver.check_inf_or_nan_grad():
        return
    gradient_clipping(params, max_norm=500)
    return

def train(args):
    # Settings
    b, c, h, w = args.batch_size, 3, 64, 64
    beta1 = 0.5
    beta2 = 0.999
    base_lr = args.learning_rate
    unpool = args.unpool

    # Context
    extension_module = args.context
    if args.context is None:
        extension_module = 'cpu'
    logger.info("Running in %s" % extension_module)
    ctx = get_extension_context(extension_module,
                                device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    # Inputs (x: lhs-image, y: rhs-image)
    x_real = nn.Variable([b, c, h, w], need_grad=False)
    y_real = nn.Variable([b, c, h, w], need_grad=False)
    z_x_real = nn.Variable([b, args.latent])
    z_y_real = nn.Variable([b, args.latent])

    x_real_test = nn.Variable([b, c, h, w], need_grad=False)
    y_real_test = nn.Variable([b, c, h, w], need_grad=False)
    z_x_real_test = nn.Variable.from_numpy_array(np.random.randn(*[b, args.latent]))
    z_y_real_test = nn.Variable.from_numpy_array(np.random.randn(*[b, args.latent]))
    
    # Models for training
    # Generate
    y_fake = G_xy(x_real, z_y_real, unpool=unpool)
    x_fake = G_yx(y_real, z_x_real, unpool=unpool)
    y_fake.persistent, x_fake.persistent = True, True
    z_x_fake = E(x_real, y_fake, args.latent)
    z_y_fake = E(x_fake, y_real, args.latent)
    # Reconstruct
    x_recon = G_yx(y_fake, z_x_fake, unpool=unpool) 
    y_recon = G_xy(x_fake, z_y_fake, unpool=unpool)
    x_recon.persistent, y_recon.persistent = True, True
    z_y_recon = E(x_real, y_fake, args.latent)
    z_x_recon = E(x_fake, y_real, args.latent)
    # Discriminate
    d_y_fake = D_y(y_fake, 64)
    d_x_fake = D_x(x_fake, 32, (3, 3))
    d_y_real = D_y(y_real, 64)
    d_x_real = D_x(x_real, 32, (3, 3))
    d_z_y_fake = D_z_x(z_y_fake, 64)  # share latent space
    d_z_x_fake = D_z_x(z_x_fake, 64)
    d_z_y_real = D_z_x(z_y_real, 64)
    d_z_x_real = D_z_x(z_x_real, 64)
    # Identity
    #x_idt = G_xy(y_real, z_y_real)
    #y_idt = G_yx(x_real, z_x_real)
    # Losses
    # Reconstruction Loss
    loss_recon_x = recon_loss(x_recon, x_real)
    loss_recon_y = recon_loss(y_recon, y_real)
    loss_recon = loss_recon_x + loss_recon_y \
                 + args.lam * (recon_loss(z_x_recon, z_x_real) \
                               + recon_loss(z_y_recon, z_y_real))
    # Generator loss (marginal matching loss)
    loss_gen_x = lsgan_loss(d_x_fake)
    loss_gen_y = lsgan_loss(d_y_fake)
    loss_gen = loss_gen_y + loss_gen_x
    # Encoder loss (marginal matching loss)
    loss_enc = lsgan_loss(d_z_y_fake) + lsgan_loss(d_z_x_fake)
    # Identity loss
    #loss_idt = recon_loss(x_real, x_idt) + recon_loss(y_real, y_idt)
    losses = 10 * loss_recon + loss_gen + loss_enc# + loss_idt * 0.5 * 10
    # Discriminator losses
    loss_dis_y = lsgan_loss(d_y_fake, d_y_real)
    loss_dis_x = lsgan_loss(d_x_fake, d_x_real)
    loss_dis_z_x = lsgan_loss(d_z_x_fake, d_z_x_real)
    loss_dis_z_y = lsgan_loss(d_z_y_fake, d_z_y_real)
    losses_dis = loss_dis_y + loss_dis_x + loss_dis_z_x + loss_dis_z_y

    # Generate
    y_fake_test = G_xy(x_real_test, z_y_real_test, unpool=unpool)
    x_fake_test = G_yx(y_real_test, z_x_real_test, unpool=unpool)
    y_fake_test.persistent, x_fake_test.persistent = True, True
    z_x_fake_test = E(x_real_test, y_fake_test, args.latent)
    z_y_fake_test = E(y_real_test, x_fake_test, args.latent)
    z_y_fake_test.persistent, z_x_fake_test.persistent = True, True
    # Reconstruct only for images
    x_recon_test = G_yx(y_fake_test, z_x_fake_test, unpool=unpool)
    y_recon_test = G_xy(x_fake_test, z_y_fake_test, unpool=unpool)
    x_fakes = [y_fake, x_fake, (z_y_fake, z_x_fake)]

    # Solvers
    solver_gen = S.Adam(base_lr, beta1, beta2)
    solver_enc = S.Adam(base_lr, beta1, beta2)
    solver_dis = S.Adam(base_lr / 5.0, beta1, beta2)
    solver_dis_z = S.Adam(base_lr / 5.0, beta1, beta2)
    with nn.parameter_scope('generator'):
        gen_params = nn.get_parameters()
        solver_gen.set_parameters(gen_params)
    with nn.parameter_scope('encoder'):
        enc_params = nn.get_parameters()
        solver_enc.set_parameters(enc_params)
    with nn.parameter_scope('discriminator'):
        dis_params = nn.get_parameters()
        solver_dis.set_parameters(dis_params)
    with nn.parameter_scope('discriminator-latent'):
        dis_params_z = nn.get_parameters()
        solver_dis_z.set_parameters(dis_params_z)

    # Datasets
    rng = np.random.RandomState(313)
    ds_train = pix2pix_data_source(
        args.dataset, train=True, paired=False, shuffle=True, rng=rng, num_data=args.num_data)
    di_train = pix2pix_data_iterator(ds_train, args.batch_size)
    ds_test = pix2pix_data_source(
        args.dataset, train=False, paired=True, shuffle=False, num_data=args.num_data)
    di_test = pix2pix_data_iterator(ds_test, args.batch_size)

    # Monitors
    monitor = Monitor(args.monitor_path)

    def make_monitor(name):
        return MonitorSeries(name, monitor, interval=1)
    monitor_loss_gen_x = make_monitor('generator_loss_x')
    monitor_loss_dis_x = make_monitor('discriminator_loss_x')
    monitor_loss_gen_y = make_monitor('generator_loss_y')
    monitor_loss_dis_y = make_monitor('discriminator_loss_y')
    monitor_loss_recon_x = make_monitor('recon_loss_x')
    monitor_loss_recon_y = make_monitor('recon_loss_y')


    def make_monitor_image(name, interval=1):
        return MonitorImage(name, monitor, interval=interval, num_images=1, 
                            normalize_method=lambda x: (x + 1.0) * 127.5)

    monitor_train_x = make_monitor_image('fake_images_train_X', 500)
    monitor_train_y = make_monitor_image('fake_images_train_Y', 500)
    monitor_train_x_recon = make_monitor_image('fake_images_X_recon_train', 500)
    monitor_train_y_recon = make_monitor_image('fake_images_Y_recon_train', 500)
    monitor_test_x = make_monitor_image('fake_images_test_X')
    monitor_test_y = make_monitor_image('fake_images_test_Y')
    monitor_test_x_recon = make_monitor_image('fake_images_recon_test_Y')
    monitor_test_y_recon = make_monitor_image('fake_images_recon_test_X')
    monitor_train_list = [
        (monitor_train_x, x_fake),
        (monitor_train_y, y_fake),
        (monitor_train_x_recon, x_recon),
        (monitor_train_y_recon, y_recon),
        (monitor_loss_gen_x, loss_gen_x), 
        (monitor_loss_dis_x, loss_dis_x), 
        (monitor_loss_gen_y, loss_gen_y), 
        (monitor_loss_dis_y, loss_dis_y),
        (monitor_loss_recon_x, loss_recon_x), 
        (monitor_loss_recon_y, loss_recon_y)
    ]
    monitor_loss_time = MonitorTimeElapsed("Elapsed Time", monitor, interval=1)
    monitor_test_list = [
        (monitor_test_x, x_fake_test),
        (monitor_test_y, y_fake_test),
        (monitor_test_x_recon, x_recon_test),
        (monitor_test_y_recon, y_recon_test)]

    # Training loop
    epoch = 0
    n_images = ds_train.size
    max_iter = args.max_epoch * n_images // args.batch_size
    for i in range(max_iter):
        # Validation
        if int((i+1) % (n_images // args.batch_size)) == 0:
            logger.info("Mode:Test,Epoch:{}".format(epoch))
            # Use training graph since there are no test mode
            x_data, y_data = di_test.next()
            x_real_test.d = x_data
            y_real_test.d = y_data
            x_recon_test.forward(clear_buffer=True)
            y_recon_test.forward(clear_buffer=True)
            # Monitor for test
            for monitor, v in monitor_test_list:
                monitor.add(i, v.d)
            monitor_loss_time.add(i)
            # Save model
            nn.save_parameters(os.path.join(
                args.monitor_path, 'params_%06d.h5' % i))
            # Learning rate decay
            for solver in [solver_gen, solver_enc, 
                           solver_dis, solver_dis_z]:
                linear_decay(solver, base_lr, epoch, args.decay_at, args.max_epoch)
            epoch += 1

        # Get data (use same data even the generaotr and encoder are updated)
        x_data, y_data = di_train.next()
        x_real.d = x_data
        y_real.d = y_data
        z_prior = np.random.randn(b, args.latent)
        z_x_real.d = z_prior
        z_y_real.d = z_prior
        # Train Generators and Encoders
        losses.forward(clear_no_need_grad=True)
        solver_gen.zero_grad()
        solver_enc.zero_grad()
        losses.backward(clear_buffer=True)
        update_paramters(solver_gen, gen_params)
        update_paramters(solver_enc, enc_params)
        #solver_gen.update()
        #solver_enc.update()

        x_data, y_data = di_train.next()
        x_real.d = x_data
        y_real.d = y_data
        z_prior = np.random.randn(b, args.latent)
        z_x_real.d = z_prior
        z_y_real.d = z_prior
        # Train discriminators
        solver_dis.zero_grad()
        solver_dis_z.zero_grad()
        losses_dis.forward(clear_no_need_grad=True)
        losses_dis.backward(clear_buffer=True)
        update_paramters(solver_dis, dis_params)
        update_paramters(solver_dis_z, dis_params_z)
        #solver_dis.update()
        #solver_dis_z.update()

        # Monitor for train
        for monitor, v in monitor_train_list:
            monitor.add(i, v.d)

    # Monitor and save
    monitor_loss_time.add(i)
    nn.save_parameters(os.path.join(
        args.monitor_path, 'params_%06d.h5' % i))
        
def main():
    args = get_args()
    save_args(args)
    train(args)


if __name__ == '__main__':
    main()




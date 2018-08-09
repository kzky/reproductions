import argparse
from nnabla.contrib.context import extension_context
from nnabla.monitor import Monitor, MonitorImage, MonitorImageTile, MonitorSeries, tile_images
from nnabla.utils.data_iterator import data_iterator
import os
import cv2

import nnabla as nn
import nnabla.functions as F
import nnabla.logger as logger
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np


class MonitorImageTileWithName(MonitorImageTile):

    def __init__(self, name, monitor, num_images=4, normalize_method=None):
        super(MonitorImageTileWithName, self).__init__(
            name, monitor, interval=10000, verbose=True, num_images=num_images, normalize_method=normalize_method)

    def add(self, name, var):
        import nnabla as nn
        from scipy.misc import imsave
        if isinstance(var, nn.Variable):
            data = var.d.copy()
        elif isinstance(var, nn.NdArray):
            data = var.data.copy()
        else:
            assert isinstance(var, np.ndarray)
            data = var.copy()
        assert data.ndim > 2
        channels = data.shape[-3]
        data = data.reshape(-1, *data.shape[-3:])
        data = data[:min(data.shape[0], self.num_images)]
        data = self.normalize_method(data)
        if channels > 3:
            data = data[:, :3]
        elif channels == 2:
            data = np.concatenate(
                [data, np.ones((data.shape[0], 1) + data.shape[-2:])], axis=1)
        tile = tile_images(data)
        path = os.path.join(self.save_dir, '{}.png'.format(name))
        imsave(path, tile)


def psnr(x, y, max_=255):
    mse = np.mean((x - y) ** 2)
    return 10 * np.log10(max_ ** 2 / mse)


def get_solver(solver):
    if solver == "Adam":
        return S.Adam

    if solver == "Momentum":
        return S.Momentum


def normalize_method(x):
    return x


def resize(x_data_s, h, w):
    x_data_s = np.asarray([cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC) \
                           for x in x_data_s.astype(np.uint8)])
    if len(x_data_s.shape) == 3:
        return x_data_s[..., np.newaxis]
    return x_data_s


def upsample(x_data_s, s):
    b, h, w, c = x_data_s.shape
    x_data_s = np.asarray([cv2.resize(x, (w * s, h * s), interpolation=cv2.INTER_CUBIC) \
                           for x in x_data_s.astype(np.uint8)])
    if len(x_data_s.shape) == 3:
        return x_data_s[..., np.newaxis]
    return x_data_s


def downsample(x_data_s, s):
    b, h, w, c = x_data_s.shape
    sh = h // s
    sw = w // s
    x_data_s = np.asarray([cv2.resize(x, (sw, sh), interpolation=cv2.INTER_CUBIC) \
                           for x in x_data_s.astype(np.uint8)])
    return x_data_s


def split(x):
    b, h, w, c = x.shape
    y = []
    for i in range(c):
        y.append(x[:, :, :, i].reshape(b, h, w, 1))
    return y


def to_BCHW(x):
    # B, H, W, C -> B, C, H, W
    return x.transpose(0, 3, 1, 2)


def to_BHWC(x):
    # B, C, H, W -> B, H, W, C
    return x.transpose(0, 2, 3, 1)


def normalize(x, de=255.0):
    return x / de


def denormalize(x, de=255.0):
    return x * de


def ycrcb_to_rgb(y, cr, cb):
    imgs = []
    imgs_ = np.concatenate([y, cr, cb], axis=3)
    for img in imgs_.astype(np.uint8):
        #print(img.dtype, np.min(img), np.max(img))
        img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2RGB)
        imgs.append(img)
    return np.asarray(imgs)



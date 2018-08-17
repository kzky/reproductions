import argparse
from nnabla.contrib.context import extension_context
from nnabla.monitor import Monitor, MonitorImage, MonitorImageTile, MonitorSeries, tile_images
from nnabla.utils.data_iterator import data_iterator
import os
from PIL import Image

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


def psnr(x, y, border=4, max_=255):
    _, _, h, w = x.shape
    x = x[:, :, border:h-border, border:w-border].astype(np.float64)
    y = y[:, :, border:h-border, border:w-border].astype(np.float64)
    mse = np.mean((x - y) ** 2)
    return 10 * np.log10(max_ ** 2 / mse)


def get_solver(solver):
    if solver == "Adam":
        return S.Adam

    if solver == "Momentum":
        return S.Momentum


def normalize_method(x):
    return x


def resize(x_data, sw, sh):
    b, h, w, c = x_data.shape
    x_data_ = []
    x_data = np.round(x_data).astype(np.uint8)
    for x in x_data:
        x = x.reshape((h, w)) if c == 1 else x.reshape((h, w, c))
        x = Image.fromarray(x).resize((sw, sh), Image.BICUBIC)
        x = np.asarray(x)
        x = x.reshape((sh, sw, c))
        x_data_.append(x)
    x_data_ = np.asarray(x_data_)
    if len(x_data_.shape) == 3:
        return x_data_[..., np.newaxis]
    return x_data_


def upsample(x_data, s):
    b, h, w, c = x_data.shape
    sh = h * s
    sw = w * s
    return resize(x_data, sw, sh)


def downsample(x_data, s):
    b, h, w, c = x_data.shape
    sh = h // s
    sw = w // s
    return resize(x_data, sw, sh)


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


def ycbcr_to_rgb(y, cb, cr):
    imgs = []
    imgs_ = np.concatenate([y, cb, cr], axis=3)
    imgs_ = np.round(imgs_).astype(np.uint8)
    for img in imgs_:
        img = Image.fromarray(img, "YCbCr").convert("RGB")
        img = np.asarray(img)
        imgs.append(img)
    return np.asarray(imgs)

def center_crop(x, s):
    _, h, w, _ = x.shape
    ch = h - h % s
    cw = w - w % s
    top = (h - ch) // 2
    bottom = (h + ch) // 2
    left = (w - cw) // 2
    right = (w + cw) // 2
    x_crop = x[:, top:bottom, left:right, :]
    return x_crop

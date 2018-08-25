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

from matlab_imresize import imresize as imresize_

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

    if solver == "Nesterov":
        return S.Nesterov


def normalize_method(x):
    return x


def imresize(x, w, h, mode="opencv"):
    if mode == "opencv":
        _, _, c0 = x.shape
        x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
        x = x.reshape((h, w, c0))
        return x
    if mode == "matlab":
        return imresize_(x, output_shape=(h, w))  # height, width
    return ValueError("Mode ({}) is not supported.".format(mode))


def resize(x_data, sw, sh, mode="opencv"):
    b, h, w, c = x_data.shape
    x_data_ = []
    for x in x_data:
        x = imresize(x, sw, sh, mode)
        x_data_.append(x)
    x_data_ = np.asarray(x_data_)
    return x_data_


def upsample(x_data, s, mode="opencv"):
    b, h, w, c = x_data.shape
    sh = h * s
    sw = w * s
    return resize(x_data, sw, sh, mode)


def downsample(x_data, s, mode="opencv"):
    
    b, h, w, c = x_data.shape
    sh = h // s
    sw = w // s
    return resize(x_data, sw, sh, mode)


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


def to_uint8(x):
    """
    Args:
      x (float or double): input image
    """
    x = np.clip(x, 0.0, 1.0)
    x = denormalize(x)
    x = np.round(x)
    x = x.astype(np.uint8)
    return x


def ycrcb_to_rgb(y, cr, cb):
    """
    Args:
      y (float or double): Y of input image.
      cr (float or double): Cr of input image.
      cb (float or double): Cb of input image.
    """

    # (B, H, W, C)
    imgs = []
    imgs_ = np.concatenate([y, cr, cb], axis=3)
    imgs_ = to_uint8(imgs_)
    for img in imgs_:
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
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


def check_grads(solvers):
    if np.prod([s.check_inf_or_nan_grad() for s in solvers]):
        return False
    else:
        return True
            

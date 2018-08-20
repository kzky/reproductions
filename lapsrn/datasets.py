from PIL import Image
import numpy as np
import io
import os
import glob
import tarfile
import zipfile

from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_iterator import data_iterator_simple
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download, get_data_home

def get_np_array_from_tar_object(tar_extractfl):
    '''converts a buffer from a tar file in np.array'''
    return np.asarray(
        bytearray(tar_extractfl.read()), 
        dtype=np.uint8)


def data_iterator_lapsrn(img_paths, batch_size=64, ih=128, iw=128, 
                         train=True, shuffle=True, rng=None):
    img_paths = [img_paths] if type(img_paths) != list else img_paths
    imgs = []
    for img_path in img_paths:
        imgs += glob.glob("{}/*.png".format(img_path))
        imgs += glob.glob("{}/*.PNG".format(img_path))
        imgs += glob.glob("{}/*.jpeg".format(img_path))
        imgs += glob.glob("{}/*.JPEG".format(img_path))


    def load_func_train(i):
        img = Image.open(imgs[i])

        # To YCbCr
        img = img.convert("YCbCr")

        # Resize
        w, h = img.size
        if h < ih or w < iw:  # ensure that size >= (ih, iw)
            img = img.resize((iw, ih),  Image.BICUBIC)
        a = np.random.uniform(0.5, 1.0)
        w, h = int(w*a), int(h*a)
        if h > ih and w > iw:
            img = img.resize((w, h),  Image.BICUBIC)
        
        # Rotate in [0, 90, 180, 270]
        k = np.random.choice([0, 90, 180, 270], size=1)
        img = img.rotate(k, expand=True) if k != 0 else img

        # To numpy
        img = np.asarray(img)

        # Crop
        h, w, _ = img.shape
        if h > ih and w > iw:
            rh = np.random.choice(np.arange(h - ih), size=1)[0]
            rw = np.random.choice(np.arange(w - iw), size=1)[0]
            img = img[rh:rh+ih, rw:rw+iw, :]

        # Flip horizontally and/or virtially
        if np.random.randint(2):
            img = img[:, ::-1, :]
        if np.random.randint(2):
            img = img[::-1, :, :]

        # img(YCrCb): [H, W, C]
        return img, None


    def load_func_test(i):
        assert batch_size == 1
        img = Image.open(imgs[i]).convert("YCbCr")
        img = np.asarray(img)
        # img(YCrCb): [H, W, C]
        return img, None


    if train:
        load_func = load_func_train
    else:
        load_func = load_func_test
    
    return data_iterator_simple(
        load_func, len(imgs), batch_size, shuffle=shuffle, rng=rng, with_file_cache=False)


if __name__ == '__main__':
    # LapSRN
    img_paths = ["/home/kzky/nnabla_data/BSDS200", 
                 "/home/kzky/nnabla_data/General100", 
                 "/home/kzky/nnabla_data/T91"]
    batch_size = 4
    train = True
    test_data = "Set5"
    di = data_iterator_lapsrn(img_paths, batch_size=batch_size, train=train)
    data = di.next()[0]
    data = di.next()[0]
    data = di.next()[0]
    print(data.shape)

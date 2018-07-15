import cv2
import numpy as np
import io
import os

from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download, get_data_home


def data_iterator_imagenet(img_path, batch_size, imsize=(256, 256), num_samples=-1, shuffle=True, rng=None):
    imgs = glob.glob("{}/*.JPEG".format(img_path))
    if num_samples == -1:
        num_samples = len(imgs)
    else:
        logger.info(
            "Num. of data ({}) is used for debugging".format(num_samples))

    def load_func(i):
        img = cv.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
        _, h, w = img.shape
        rh, rw = h - imsize[0], w - imsize[1]
        ry = np.random.choice(np.arrange(rh), 1)
        rx = np.random.choice(np.arrange(rw), 1)
        img = img[:, ry:ry + imsize[0], rx:rx + imsize[1]]
        return img, None

    return data_iterator_simple(load_func, num_samples, batch_size, shuffle=shuffle, rng=rng, with_file_cache=False)


def load_kodak():
    image_uri = "http://r0k.us/graphics/kodak/kodak/"
    logger.info('Getting image data from {}.'.format(image_uri))
    images = []
    dirpath = os.path.join(get_data_home(), "kodak")
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    for i in range(0, 24):
        filename = "kodim{:02d}.png".format(i + 1)
        fpath = os.path.join(dirpath, filename)
        print(fpath)
        if not os.path.exists(fpath):
            r = download(os.path.join(image_uri, filename), fpath)
        img = cv2.imread(fpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
        images.append(img)
        
    logger.info('Getting image data done.')
    return images, None


class KodakDataSource(DataSource):
    '''
    Get data directly from Kodak dataset from Internet.
    '''

    def _get_data(self, position):
        image = self._images[position]
        return image, None

    def __init__(self, ):
        super(KodakDataSource, self).__init__()
        self._images, _ = load_kodak()
        self._size = len(self._images)
        self._variables = ('x', )
        self.rng = np.random.RandomState(313)
        self.reset()

    def reset(self):
        pass

def data_iterator_kodak(batch_size=1):
    '''
    '''
    return data_iterator(KodakDataSource(), batch_size=1)
                         

if __name__ == '__main__':
    di = data_iterator_kodak()
    img = di.next()

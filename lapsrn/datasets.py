import cv2
import numpy as np
import io
import os
import glob
import tarfile

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
"""
Imagenet
"""
def data_iterator_imagenet(img_path, batch_size, imsize=(256, 256), num_samples=-1, shuffle=True, rng=None):
    imgs = glob.glob("{}/*.JPEG".format(img_path))
    if num_samples == -1:
        num_samples = len(imgs)
    else:
        logger.info(
            "Num. of data ({}) is used for debugging".format(num_samples))

    def load_func(i):
        #TODO: try to use F.image_augmentation
        img = cv2.imread(imgs[i])
        img = cv2.resize(img, imsize)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))

        # Paper said 256x256 random crop, but some images do not have 256 height or width.
        # _, h, w = img.shape
        # rh, rw = h - imsize[0], w - imsize[1]
        # ry = np.random.choice(np.arrange(rh), 1)
        # rx = np.random.choice(np.arrange(rw), 1)
        # img = img[:, ry:ry + imsize[0], rx:rx + imsize[1]]
        
        return img, None

    return data_iterator_simple(load_func, num_samples, batch_size, shuffle=shuffle, rng=rng, with_file_cache=False)


"""
Kodak
"""
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
    return images


class KodakDataSource(DataSource):
    '''
    Get data directly from Kodak dataset from Internet.
    '''

    def _get_data(self, position):
        image = self._images[position]
        return np.asarray(image), None

    def __init__(self, ):
        super(KodakDataSource, self).__init__()
        self._images = load_kodak()
        self._size = len(self._images)
        self._variables = ('x', )
        self.rng = np.random.RandomState(313)
        self.reset()

    def reset(self):
        pass


def data_iterator_kodak():
    '''
    '''
    return data_iterator(KodakDataSource(), batch_size=1)
                         

"""
BSDS{300, 500}
"""

def load_bsds300(dataset="bsds300"):
    if dataset == "bsds300":
        data_uri = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
    elif dataset == "bsds500":
        data_uri = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
    else:
        raise ValueError("{} is not supported".format(dataset))

    logger.info('Getting data from {}.'.format(data_uri))
    r = download(data_uri)  # file object returned
    with tarfile.open(fileobj=r, mode="r:gz") as tar:
        images = []
        for member in tar.getmembers():
            if ".jpg" not in member.name:
                continue
            fp = tar.extractfile(member)
            img = cv2.imdecode(get_np_array_from_tar_object(fp), 3)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
            images.append(img)
    return images


class BSDSDataSource(DataSource):
    '''
    Get data directly from Kodak dataset from Internet.
    '''

    def _get_data(self, position):
        image = self._images[position]
        return np.asarray(image), None

    def __init__(self, dataset="bsds300"):
        super(BSDSDataSource, self).__init__()
        if dataset == "bsds300":
            self._images = load_bsds300()
        elif dataset == "bsds500":
            self._images = load_bsds500()
        else:
            raise ValueError("{} is not supported".format(dataset))
        self._size = len(self._images)
        self._variables = ('x', )
        self.rng = np.random.RandomState(313)
        self.reset()

    def reset(self):
        pass


def data_iterator_bsds(dataset="bsds300"):
    return data_iterator(BSDSDataSource(), batch_size=1)


"""
Set{5, 14}
"""
def load_set(dataset="set14"):
    pass

class SetDataSource(DataSource):
    '''
    Get data directly from Set dataset from Internet.
    '''
    pass

def data_iterator_set():
    '''
    '''
    return data_iterator(SetDataSource(), batch_size=1)


if __name__ == '__main__':
    # Kodak
    di = data_iterator_kodak()
    img = di.next()

    # BSDS300
    di = data_iterator_bsds(dataset="bsds300")
    img = di.next()

    # BSDS500
    di = data_iterator_bsds(dataset="bsds500")
    img = di.next()
    

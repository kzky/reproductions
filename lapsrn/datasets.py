import cv2
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
LapSRN
"""

def load_lapsrn(train=True, test_data="Set5"):
    if train:
        data_uri = "http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_training_datasets.zip"
    else:
        data_uri = "http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_testing_datasets.zip"
    logger.info('Getting image data from {}.'.format(data_uri))
    images = []
    r = download(data_uri)
    with zipfile.ZipFile(io.BytesIO(r.read())) as fp:
        for name in fp.namelist():
            if not name.endswith(".png"):
                continue
            if not train and test_data not in name:
                continue
            print(name)
            data = fp.read(name)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), 3)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
            images.append(img)
    logger.info('Getting image data done.')
    r.close()
    return images


class LapSRNDataSource(DataSource):
    '''
    Get data directly from Set dataset from Internet.
    '''
    
    def _get_data(self, position):
        image = self._images[position]
        return image, None

    def __init__(self, train=True, shuffle=True, test_data="Set5"):
        super(LapSRNDataSource, self).__init__()
        self._images = load_lapsrn(train=train, test_data=test_data)
        self._size = len(self._images)
        self._variables = ('x', )
        self.rng = np.random.RandomState(313)
        self.reset()

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(LapSRNDataSource, self).reset()


def data_iterator_lapsrn(img_paths, batch_size=64, ih=128, iw=128, 
                         train=True, shuffle=True, rng=None):
    img_paths = [img_paths] if type(img_paths) != list else img_paths
    imgs = []
    for img_path in img_paths:
        imgs += glob.glob("{}/*.png".format(img_path))

    def load_func_train(i):
        img = cv2.imread(imgs[i])

        # Resize in [0.5 1.0]  #TODO: this is really need?
        h, w, c = img.shape
        if h < ih or w < iw:
            img = cv2.resize(img, (ih, iw))
        a = np.random.uniform(0.5, 1.0)
        h, w = int(h*a), int(w*a)
        if h > ih and w > iw:
            img = cv2.resize(img, (h, w))
        
        # To RGB-(c, h, w) array
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))

        # Crop
        c, h, w = img.shape
        if h > ih and w > iw:
            rh = np.random.choice(np.arange(h - ih), size=1)[0]
            rw = np.random.choice(np.arange(w - iw), size=1)[0]
            img = img[:, rh:rh+ih, rw:rw+iw]

        # Rotate in [0, 90, 180, 270]
        k = np.random.choice([0, 90, 180, 270], size=1)
        img = np.rot90(img, k=k, axes=(1, 2))

        # Flip
        if np.random.randint(2):
            img = img[:, :, ::-1]

        return img, None

    def load_func_test(i):
        assert batch_size == 1
        img = cv2.imread(imgs[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
        return img, None            

    if train:
        load_func = load_func_train
    else:
        load_func = load_func_test
    
    return data_iterator_simple(
        load_func, len(imgs), batch_size, shuffle=shuffle, rng=rng, with_file_cache=False)


if __name__ == '__main__':
    # LapSRN
    img_paths = ["/home/kzky/nnabla_data/BSD200", 
                 "/home/kzky/nnabla_data/General100", 
                 "/home/kzky/nnabla_data/T90"]
    batch_size = 4
    train = True
    test_data = "Set5"
    di = data_iterator_lapsrn(img_paths, batch_size=batch_size, train=train)
    data = di.next()[0]
    data = di.next()[0]
    data = di.next()[0]
    print(data.shape)

"""
Provide data iterator for horse2zebra examples.
"""

import os
import scipy.misc
import zipfile
from contextlib import contextmanager
import numpy as np
import nnabla as nn
from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download, get_data_home

import cv2
import tarfile

def get_np_array_from_tar_object(tar_extractfl):
    '''converts a buffer from a tar file in np.array'''
    return np.asarray(
        bytearray(tar_extractfl.read()),
        dtype=np.uint8)

def load_pix2pix_dataset(dataset="edges2shoes", train=True, 
                         normalize_method=lambda x: (x - 127.5) / 127.5, 
                         num_data=-1):
    image_uri = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/{}.tar.gz'\
                .format(dataset)
    logger.info('Getting {} data from {}.'.format(dataset, image_uri))
    r = download(image_uri)

    # Load concatenated images, then save separately
    #TODO: how to do for test
    img_A_list = []
    img_B_list = []

    with tarfile.open(fileobj=r, mode="r") as tar:
        cnt = 0
        for tinfo in tar.getmembers():
            if not ".jpg" in tinfo.name:
                continue
            if not ((train==True and "train" in tinfo.name) \
                    or (train==False and "val" in tinfo.name)):
                continue

            logger.info("Loading {} ... at {}".format(tinfo.name, cnt))
            f = tar.extractfile(tinfo)
            #img = scipy.misc.imread(f, mode="RGB")
            img = cv2.imdecode(get_np_array_from_tar_object(f), 3)
            h, w, c = img.shape
            img_A = img[:, 0:w // 2, :]
            img_B = img[:, w // 2:, :]
            img_A = cv2.resize(img_A, (64, 64), interpolation=cv2.INTER_CUBIC)
            img_B = cv2.resize(img_B, (64, 64), interpolation=cv2.INTER_CUBIC)
            img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
            img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
            
            img_A_list.append(img_A)
            img_B_list.append(img_B)
            cnt += 1

            if num_data != -1 and cnt >= num_data:
                break

    r.close()
    logger.info('Getting image data done.')
    return np.asarray(img_A_list), np.asarray(img_B_list)

#TODO: abstract this data source under the autmented-cycle gan dataset
# adding size of images for addressing different shapes of different datasets is enough?
class Edges2ShoesDataSource(DataSource):
    
    def _get_data(self, position):
        if self._paired:
            images_A = self._images_A[self._indices[position]]
            images_B = self._images_B[self._indices[position]]
        else:
            images_A = self._images_A[self._indices_A[position]]
            images_B = self._images_B[self._indices_B[position]]
        return self._normalize_method(images_A), self._normalize_method(images_B)

    def __init__(self, dataset="edges2shoes", train=True, paired=True, 
                 shuffle=False, rng=None, 
                 normalize_method=lambda x: (x - 127.5) / 127.5, 
                 num_data=-1):
        super(Edges2ShoesDataSource, self).__init__(shuffle=shuffle)

        if rng is None:
            rng = np.random.RandomState(313)

        images_A, images_B = load_pix2pix_dataset(dataset=dataset, train=train, 
                                                  normalize_method=normalize_method, 
                                                  num_data=num_data)
        self._images_A = images_A
        self._images_B = images_B
        self._size = len(self._images_A)  # since A and B is the paired image, lengths are the same
        self._size_A = len(self._images_A)
        self._size_B = len(self._images_B)
        self._variables = ('A', 'B')
        self._paired = paired
        self._normalize_method = normalize_method
        self.rng = rng
        self.reset()

    def reset(self):
        if self._paired:
            self._indices = self.rng.permutation(self._size) \
                            if self._shuffle else np.arange(self._size)
                               
        else:
            self._indices_A = self.rng.permutation(self._size_A) \
                              if self._shuffle else np.arange(self._size_A)
            self._indices_B = self.rng.permutation(self._size_B) \
                              if self._shuffle else np.arange(self._size_B)
        return super(Edges2ShoesDataSource, self).reset()
    

def pix2pix_data_source(dataset, train=True, paired=True, shuffle=False, rng=None, num_data=-1):
    return Edges2ShoesDataSource(dataset=dataset,
                                 train=train, paired=paired, shuffle=shuffle, rng=rng,
                                 num_data=num_data)

def pix2pix_data_iterator(data_source, batch_size):
    return data_iterator(data_source,
                         batch_size=batch_size,
                         with_memory_cache=False,
                         with_file_cache=False)

if __name__ == '__main__':
    # Hand-made test
    
    dataset = "edges2shoes"
    num_data = 100
    ds = Edges2ShoesDataSource(dataset=dataset, num_data=num_data)
    di = pix2pix_data_iterator(ds, batch_size=1)
    

    
    

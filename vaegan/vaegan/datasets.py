import numpy as np
import os
from chainer import cuda
import glob
import cv2

class MNISTDataReader(object):
    """DataReader
    """
    def __init__(self,
                     train_path=\
                     "/home/kzk/.chainer/dataset/pfnet/chainer/mnist/train.npz",
                     test_path=\
                     "/home/kzk/.chainer/dataset/pfnet/chainer/mnist/test.npz",
                     batch_size=64):

        # Concate data    
        train_data = dict(np.load(train_path))
        test_data = dict(np.load(test_path))
        self.data = np.vstack([train_data["x"], test_data["x"]])

        self._batch_size = batch_size
        self._next_position = 0
        self._n_data = len(self.data)

        print("Num. of samples {}".format(self._n_data))
        
    def get_train_batch(self):
        # Read and reshape data
        beg = self._next_position
        end = self._next_position + self._batch_size
        batch_data = self.data[beg:end, :]
        bs = batch_data.shape[0]
        batch_data = ((batch_data -127.5) / 127.5)\
            .astype(np.float32)  #.reshape(bs, 28, 28)

        # Reset pointer
        self._next_position += self._batch_size
        if self._next_position >= self._n_data:
            self._next_position = 0

            # Shuffle
            idx = np.arange(self._n_data)
            np.random.shuffle(idx)
            self.data = self.data[idx]
            
        return batch_data


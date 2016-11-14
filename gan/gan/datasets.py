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
        batch_data = (batch_data / 255.).astype(np.float32).reshape(bs, 28, 28)

        # Reset pointer
        self._next_position += self._batch_size
        if self._next_position >= self._n_data:
            self._next_position = 0

            # Shuffle
            idx = np.arange(self._n_data)
            np.random.shuffle(idx)
            self.data = self.data[idx]
            
        return batch_data

class LFWDataReader(object):
    """DataReader
    """
    def __init__(self,
                     train_path=\
                     "/home/kzk/datasets/lfw",
                     batch_size=64):

        # Read and concate data
        data = []
        fpaths = glob.glob(os.path.join(train_path, "*", "*"))
        for fpath in fpaths:
            img = cv2.imread(fpath)

            # Center cropping
            img = img[128-64:128+64, 128-64:128+64, :]
            data.append(img)
        self.data = np.array(data, dtype=np.uint8)
        self._batch_size = batch_size
        self._next_position = 0
        self._n_data = len(self.data)

        print("Num. of samples {}".format(self._n_data))
        
    def get_train_batch(self,):
        # Read and reshape data
        beg = self._next_position
        end = self._next_position + self._batch_size
        batch_data = self.data[beg:end, :]
        batch_data = (batch_data.astype(np.float32) - 127.5) / 127.5 # Scale (-1, 1) for tanh.
        batch_data_ = []

        for data in batch_data:
            data_ = cv2.resize(data, (64, 64)).transpose(2, 0, 1)  # 128x128 to 64x64
            batch_data_.append(data_)
        batch_data = np.array(batch_data_)

        # Reset pointer
        self._next_position += self._batch_size
        if self._next_position >= self._n_data:
            self._next_position = 0

            # Shuffle
            idx = np.arange(self._n_data)
            np.random.shuffle(idx)
            self.data = self.data[idx]
        
        return batch_data


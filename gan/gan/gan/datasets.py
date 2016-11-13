import numpy as np
import os
from chainer import cuda

class MNISTDataReader(object):
    """DataReader
    """
    def __init__(self,
        l_train_path=\
            "/home/kzk/.chainer/dataset/pfnet/chainer/mnist/l_train.npz",
        u_train_path=\
            "/home/kzk/.chainer/dataset/pfnet/chainer/mnist/u_train.npz", 
        test_path="/home/kzk/.chainer/dataset/pfnet/chainer/mnist/test.npz",
        batch_size=32,
        n_cls=10):
            
        self.l_train_data = dict(np.load(l_train_path))
        self.u_train_data = dict(np.load(u_train_path))
        self.test_data = dict(np.load(test_path))

        self._batch_size = batch_size
        self._next_position_l_train = 0
        self._next_position_u_train = 0

        self._n_l_train_data = len(self.l_train_data["x"])
        self._n_u_train_data = len(self.u_train_data["x"])
        self._n_test_data = len(self.test_data["x"])
        self._n_cls = n_cls

        print("Num. of labeled samples {}".format(self._n_l_train_data))
        print("Num. of unlabeled samples {}".format(self._n_u_train_data))
        print("Num. of test samples {}".format(self._n_test_data))
        print("Num. of classes {}".format(self._n_cls))
        
    def get_l_train_batch(self,):
        """Return next batch data.

        Return next batch data. Once all samples are read, permutate all samples.
        
        Returns
        ------------
        tuple of 2: First is for sample and the second is for label.
                            First data is binarized if a value is greater than 0, then 1;
                            otherwise 0.
        """
        # Read data
        beg = self._next_position_l_train
        end = self._next_position_l_train+self._batch_size
        batch_data_x_ = self.l_train_data["x"][beg:end, :]
        batch_data_y_ = self.l_train_data["y"][beg:end]
        batch_data_x = (batch_data_x_ / 255.).astype(np.float32)
        batch_data_y = batch_data_y_.astype(np.int32)

        # Reset pointer
        self._next_position_l_train += self._batch_size
        if self._next_position_l_train >= self._n_l_train_data:
            self._next_position_l_train = 0

            # Shuffle
            idx = np.arange(self._n_l_train_data)
            np.random.shuffle(idx)
            self.l_train_data["x"] = self.l_train_data["x"][idx]
            self.l_train_data["y"] = self.l_train_data["y"][idx]

        
        return batch_data_x, batch_data_y

    def get_u_train_batch(self,):
        """Return next batch data.

        Return next batch data. Once all samples are read, permutate all samples.

        Returns:
        tuple of 2: First is for sample and the second is for label.
                            First data is binarized if a value is greater than 0, then 1;
                            otherwise 0.
        """
        # Read data
        beg = self._next_position_u_train
        end = self._next_position_u_train+self._batch_size
        batch_data_x_ = self.u_train_data["x"][beg:end, :]
        batch_data_y_ = self.u_train_data["y"][beg:end]
        batch_data_x = (batch_data_x_ / 255.).astype(np.float32)
        batch_data_y = batch_data_y_.astype(np.int32)

        # Reset pointer
        self._next_position_u_train += self._batch_size
        if self._next_position_u_train >= self._n_u_train_data:
            self._next_position_u_train = 0

            # Shuffle
            idx = np.arange(self._n_u_train_data)
            np.random.shuffle(idx)
            self.u_train_data["x"] = self.u_train_data["x"][idx]
            self.u_train_data["y"] = self.u_train_data["y"][idx]
        
        return batch_data_x, batch_data_y

    def get_test_batch(self,):
        """Return next batch data.

        Returns
        ------------
        tuple of 2: First is for sample and the second is for label.
                            First data is binarized if a value is greater than 0, then 1;
                            otherwise 0.
        """

        # Read data
        batch_data_x_ = self.test_data["x"]
        batch_data_y_ = self.test_data["y"]
        batch_data_x = (batch_data_x_ / 255.).astype(np.float32)
        batch_data_y = batch_data_y_.astype(np.int32)

        return batch_data_x , batch_data_y

class Separator(object):
    """Seprate the original samples to labeled samples and unlabeled samples.

    Seprate the original samples to labeled samples and unlabeled samples in such
    way; the number of labeled samples are selected randomly, it is equal to `l`, 
    and the others are unlabeled samples.
    """

    def __init__(self, l=100):
        self.l = l

    def separate_then_save(self,
                fpath="/home/kzk/.chainer/dataset/pfnet/chainer/mnist/train.npz"):
        ldata, udata = self._separate(fpath)
        self._save_ssl_data(fpath, ldata, udata)
        
    def _separate(self,
                 fpath="/home/kzk/.chainer/dataset/pfnet/chainer/mnist/train.npz"):
        
        data = np.load(fpath)
        n = len(data["x"])
        idxs = np.arange(n)
        idxs_l = self._sample_indices(data["y"])
        idxs_u = np.asarray(list(set(idxs) - set(idxs_l)))

        ldata = {}
        udata = {}
        ldata["x"] = data["x"][idxs_l]
        ldata["y"] = data["y"][idxs_l]
        udata["x"] = data["x"][idxs_u]
        udata["y"] = data["y"][idxs_u]

        return ldata, udata

    def _sample_indices(self, y):
        classes = set(y)
        indicies = []
        n_for_each_classes = int(1. * self.l / len(classes))
        for c in classes:
            indices_for_c = np.where(y==c)[0]
            indicies += np.random.choice(indices_for_c, n_for_each_classes,
                                            replace=False).tolist()
        return indicies
        
    def _save_ssl_data(self, fpath, ldata, udata):
        dpath = os.path.dirname(fpath)
        fname = os.path.basename(fpath)

        l_fname = "l_{}".format(fname)
        u_fname = "u_{}".format(fname)
        
        ldata_fpath = os.path.join(dpath, l_fname)
        udata_fpath = os.path.join(dpath, u_fname)

        np.savez(ldata_fpath, **ldata)
        np.savez(udata_fpath, **udata)

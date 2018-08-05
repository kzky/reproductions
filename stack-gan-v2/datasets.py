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

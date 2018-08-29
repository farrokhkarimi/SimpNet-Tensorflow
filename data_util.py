# ========================================
# [] File Name : data_util.py
#
# [] Creation Date : Aug 2018
#
# [] Author : Ali Gholami
#
# ========================================

import numpy as np
import tensorflow as tf
import os

def load_image_data(dir_path, batch_size):

    train_data = np.array([1])
    test_data = np.array([2])

    return train_data, test_data


def safe_mkdir(path):

    try:
        os.mkdir(path)
    except OSError as err:
        pass




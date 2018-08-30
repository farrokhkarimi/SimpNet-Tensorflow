# ========================================
# [] File Name : data_util.py
#
# [] Creation Date : Aug 2018
#
# [] Author : Ali Gholami
#
# ========================================

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import struct

def parse_data(path, dataset, flatten):
    if dataset != 'train' and dataset != 'test':
        raise NameError('dataset must be train or test')

    label_file = os.path.join(path, dataset + '-labels-idx1-ubyte')
    with open(label_file, 'rb') as file:
        _, num = struct.unpack(">II", file.read(8))
        labels = np.fromfile(file, dtype=np.int8) #int8
        new_labels = np.zeros((num, 10))
        new_labels[np.arange(num), labels] = 1
    
    img_file = os.path.join(path, dataset + '-images-idx3-ubyte')
    with open(img_file, 'rb') as file:
        _, num, rows, cols = struct.unpack(">IIII", file.read(16))
        imgs = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols) #uint8
        imgs = imgs.astype(np.float32) / 255.0
        if flatten:
            imgs = imgs.reshape([num, -1])

    return imgs, new_labels


def get_nih_data(dir_path, train_val_split=0.7):

    imgs, labels = parse_data(dir_path, 'train', flatten=False)

    ds_count = labels.shape[0]
    train_count = ds_count * train_val_split
    val_count = ds_count - train_count

    indices = np.random.permutation(ds_count)

    train_idx, val_idx = indices[:train_count], indices[train_count:]

    train_data, train_labels = imgs[train_idx, :], labels[train_idx, :]
    val_data, val_labels = imgs[val_idx, :], labels[val_idx, :]

    return (train_data, train_labels), (val_data, val_labels)

def get_image_dataset(dir_path, batch_size, split=0.7):

    train_data, val_data = get_nih_data(dir_path, split)

    # Create the dataset for our train data
    train_data = tf.data.Dataset.from_tensor_slices(train_data)
    train_data = train_data.batch(batch_size)

    # Create the dataset for our test data
    val_data = tf.data.Dataset.from_tensor_slices(val_data)
    val_data = val_data.batch(batch_size)

    return train_data, val_data


def safe_mkdir(path):

    try:
        os.mkdir(path)
    except OSError as err:
        pass




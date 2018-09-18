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
from glob import glob
import struct
import matplotlib.pyplot as plt
import cv2

def parse_data(path, dataset, flatten):
    if dataset != 'train' and dataset != 'validation':
        raise NameError('dataset must be train or validation.')

    # Load the data
    DATA_ROOT = 'data/'
    train_list = pd.read_csv(DATA_ROOT + 'Data_Entry_2017.csv')

    # Scan the directory of images
    train_images = {os.path.basename(x): x for x in 
                    glob(os.path.join(DATA_ROOT, 'images*', '*.png'))}

    print('Scans found:', len(train_images), ', Total Headers', train_list.shape[0])

    # Convert the labels into the binary format
    n_classes = train_list['Finding Labels'].value_counts()[:15]
    # print("Number of classes {0}".format(n_classes))
    fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
    ax1.bar(np.arange(len(n_classes))+0.5, n_classes)
    ax1.set_xticks(np.arange(len(n_classes))+0.5)
    _ = ax1.set_xticklabels(n_classes.index, rotation = 90)
    # plt.show()

    train_list['Finding Labels'] = train_list['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
    from itertools import chain
    all_labels = np.unique(list(chain(*train_list['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    all_labels = [x for x in all_labels if len(x)>0]
    # print('All Labels ({}): {}'.format(len(all_labels), all_labels))
    for c_label in all_labels:
        if len(c_label)>1: # leave out empty labels
            train_list[c_label] = train_list['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

    # print(train_list.columns)

    # since the dataset is very unbiased, we can resample it to be a more reasonable collection
    # weight is 0.1 + number of findings
    # sample_weights = train_list['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 4e-2
    # sample_weights /= sample_weights.sum()
    # train_list = train_list.sample(500, weights=sample_weights)
    # label_counts = train_list['Finding Labels'].value_counts()[:15]
    # label_counts = 100*np.mean(train_list[all_labels].values,0)
    
    # train_list['disease_vec'] = train_list.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])

    new_labels = train_list[all_labels]


    list_of_imgs = []
    img_dir = "./data/images/"
    for img in os.listdir(img_dir):
        img = os.path.join(img_dir, img)
        if not img.endswith(".png"):
            continue
        a = cv2.imread(img)
        if a is None:
            print("Unable to read image", img)
            continue
        list_of_imgs.append(a.flatten())
    imgs = np.array(list_of_imgs)

    print("Dimensionality of images: {0}".format(imgs.shape))
    return imgs, new_labels.as_matrix()


def get_nih_data(dir_path, train_val_split=0.7):

    imgs, labels = parse_data(dir_path, 'train', flatten=False)

    imgs = imgs.astype(np.float32, copy=False)

    ds_count = labels.shape[0]
    train_count = int(ds_count * train_val_split)
    val_count = int(ds_count - train_count)

    indices = np.random.permutation(ds_count)

    train_idx, val_idx = indices[:train_count], indices[train_count:]

    train_data, train_labels = imgs[train_idx, :], labels[train_idx]
    val_data, val_labels = imgs[val_idx, :], labels[val_idx]

    print("Dimensionality of train labels: {0}".format(train_labels.shape))
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

print(parse_data('..', 'train', flatten=False))
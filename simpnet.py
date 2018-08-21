# ========================================
# [] File Name : simpnet.py
#
# [] Creation Date : Aug 2018
#
# [] Author : Ali Gholami
#
# ========================================

import tensorlfow as tf
from cnn_util import conv_relu, maxpool, fully_connected
from cnn_config import *

class SimpNet(object):

    def __init__(self):

        # Dropout rate
        self.keep_prob = 0.7

        # Learning rate
        self.learning_rate = 0.001

        # Data pipeline
        self.imgs = None

    
    def inference(self):

        conv1 = conv_relu(
            inputs=self.imgs,
            filters=CONV1_NUM_FILTERS,
            k_size=CONV1_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_1'
        )

        conv2 = conv_relu(
            inputs=conv1,
            filters=CONV2_NUM_FILTERS,
            k_size=CONV2_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_2'
        )

        conv3 = conv_relu(
            inputs=conv2,
            filters=CONV3_NUM_FILTERS,
            k_size=CONV3_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_3'
        )

        conv4 = conv_relu(
            inputs=conv3, 
            filters=CONV4_NUM_FILTERS,
            k_size=CONV4_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_4'
        )

        conv5 = conv_relu(
            inputs=conv4,
            filters=CONV5_NUM_FILTERS,
            k_size=CONV5_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_5'
        )

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

        pool1 = maxpool(
            inputs=conv5,
            k_size=MAXPOOL1_SIZE,
            stride=1,
            padding='VALID',
            scope_name='pool_1'
        )

        conv6 = conv_relu(
            inputs=pool1,
            filters=CONV6_NUM_FILTERS,
            k_size=CONV6_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_6'
        )

        conv7 = conv_relu(
            inputs=conv6,
            filters=CONV7_NUM_FILTERS,
            k_size=CONV7_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_7'
        )

        conv8 = conv_relu(
            inputs=conv7,
            filters=CONV8_NUM_FILTERS,
            k_size=CONV8_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_8'
        )

        conv9 = conv_relu(
            inputs=conv8,
            filters=CONV9_NUM_FILTERS,
            k_size=CONV9_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_9'
        )

        conv10 = conv_relu(
            inputs=conv9,
            filters=CONV10_NUM_FILTERS,
            k_size=CONV10_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_10'
        )

        pool2 = maxpool(
            inputs=conv10,
            k_size=MAXPOOL2_SIZE,
            stride=1,
            padding='VALID',
            scope_name='pool_2'
        )

        conv11 = conv_relu(
            inputs=pool2,
            filters=CONV11_NUM_FILTERS,
            k_size=CONV11_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_11'
        )

        conv12 = conv_relu(
            inputs=conv11,
            filters=CONV11_NUM_FILTERS,
            k_size=CONV11_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_12',
        )

        conv13 = conv_relu(
            inputs=conv12, 
            filters=CONV13_NUM_FILTERS,
            k_size=CONV13_FILTER_SIZE,
            stride=1,
            padding='SAME',
            scope_name='conv_13'
        )

        global_pool = maxpool(
            inputs=conv13,
            k_size=MAXPOOL3_SIZE,
            stride=1,
            padding='VALID',
            scope_name='global_pool'
        )
        
        
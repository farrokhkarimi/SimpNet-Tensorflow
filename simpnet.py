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

        conv_relu(
            inputs=self.imgs,
            filters=,
            k_size=C1_FILTER_SIZE


        )

        
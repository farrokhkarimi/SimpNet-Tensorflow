# ========================================
# [] File Name : simpnet.py
#
# [] Creation Date : Aug 2018
#
# [] Author : Ali Gholami
#
# ========================================

import tensorlfow as tf

class SimpNet(object):

    def __init__(self):

        # Dropout rate
        self.keep_prob = 0.7

        # Learning rate
        self.learning_rate = 0.001

    
    def inference(self):
        
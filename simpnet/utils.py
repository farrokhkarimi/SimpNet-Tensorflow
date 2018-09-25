# ========================================
# [] File Name : cnn_config.py
#
# [] Creation Date : Aug 2018
#
# [] Author : Ali Gholami
#
# =======================================

import os

def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

# ========================================
# [] File Name : cnn_config.py
#
# [] Creation Date : Aug 2018
#
# [] Author : Ali Gholami
#
# =======================================

import os
import random

def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def shuffle_csv(path):
    """ Shuffle data lines inside a csv and return the new file path"""

    with open(path, 'r') as source:
        data = [(random.random(), line) for line in source]

    # Shuffle 
    data.sort()

    root, ext = os.path.splitext(path)

    shuffled_csv = root + '_shuffled' + ext
    
    # Write the new file
    with open(shuffled_csv, 'w') as target:
        for _, line in data:
            target.write(line)

    return shuffled_csv
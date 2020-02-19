import os
import sys
import time
import argparse
import random

import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    print('cannot import matplotlib!')

from utils import get_data
from collections import defaultdict
from sketch_utils import random_hash, random_hash_avg, random_hash_with_sign

def count_min(y, n_buckets, n_hash, loss_function):
    if len(y) == 0:
        return 0    # avoid division of 0

    counts_all = np.zeros((n_hash, n_buckets))
    y_buckets_all = np.zeros((n_hash, len(y)), dtype=int)
    for i in range(n_hash):
        counts, loss, y_buckets = random_hash(y, n_buckets)
        counts_all[i] = counts
        y_buckets_all[i] = y_buckets

    loss = 0
    for i in range(len(y)):
        y_est = np.min([counts_all[k, y_buckets_all[k, i]] for k in range(n_hash)])
        loss += loss_function(y[i], y_est)
        
    return loss / len(y)

def count_sketch(y, n_buckets, n_hash, loss_function):
    if len(y) == 0:
        return 0    # avoid division of 0

    counts_all = np.zeros((n_hash, n_buckets))
    y_buckets_all = np.zeros((n_hash, len(y)), dtype=int)
    y_signs_all = np.zeros((n_hash, len(y)), dtype=int)
    for i in range(n_hash):
        counts, y_buckets, y_signs = random_hash_with_sign(y, n_buckets)
        counts_all[i] = counts
        y_buckets_all[i] = y_buckets
        y_signs_all[i] = y_signs

    loss = 0
    for i in range(len(y)):
        y_est = np.median(
            [y_signs_all[k, i] * counts_all[k, y_buckets_all[k, i]] for k in range(n_hash)])
        loss += loss_function(y[i], y_est)
    return loss / len(y)



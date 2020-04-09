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
from sketch_utils import random_hash, random_hash_avg, random_hash_with_sign, random_hash_with_bucket_weights

def count_min(y, n_buckets, n_hash, loss_function=None):
    if len(y) == 0:
        return 0    # avoid division of 0

    counts_all = np.zeros((n_hash, n_buckets))
    y_buckets_all = np.zeros((n_hash, len(y)), dtype=int)
    for i in range(n_hash):
        counts, loss, y_buckets = random_hash(y, n_buckets)
        counts_all[i] = counts
        y_buckets_all[i] = y_buckets

    loss = 0
    y_est = np.zeros(len(y))
    for i in range(len(y)): 
        sketch_estimates = [counts_all[k, y_buckets_all[k, i]] for k in range(n_hash)]
        y_est[i] = np.min(sketch_estimates)

        if loss_function != None:
            loss += loss_function(y[i], y_est[i])

        # if i < 10:
        #   print("(sketch) real: " + str(int(y[i])) + " est: " + str(int(y_est[i])) + " delta: " + str(int(np.abs(y[i] - y_est[i]))))
        #   print("real prob: " + str(y_prob[i-start]) + " pred prob: " + str(scaling_factor*y_pred[i - start]))


    return y_est, loss/len(y)


def count_min_with_weights(y, n_buckets, n_hash, loss_function=None, weights=None):
    if len(y) == 0:
        return 0    # avoid division of 0

    counts_all = np.zeros((n_hash, n_buckets))
    y_buckets_all = np.zeros((n_hash, len(y)), dtype=int)
    for i in range(n_hash):
        counts, weights, y_buckets = random_hash_with_bucket_weights(y, n_buckets, weights=weights)
        counts_all[i] = counts
        y_buckets_all[i] = y_buckets

    loss = 0
    y_est = np.zeros(len(y))
    for i in range(len(y)): 
        sketch_estimates = [counts_all[k, y_buckets_all[k, i]] for k in range(n_hash)]
        y_est[i] = np.min(sketch_estimates)

        if loss_function != None:
            loss += loss_function(y[i], y_est[i])

        # if i < 10:
        #   print("(sketch) real: " + str(int(y[i])) + " est: " + str(int(y_est[i])) + " delta: " + str(int(np.abs(y[i] - y_est[i]))))
        #   print("real prob: " + str(y_prob[i-start]) + " pred prob: " + str(scaling_factor*y_pred[i - start]))


    return y_est, loss/len(y)


def count_sketch(y, n_buckets, n_hash, loss_function=None, estimates=None):
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
    y_est = np.zeros(len(y))
    for i in range(len(y)):
        sketch_estimates = [y_signs_all[k, i] * counts_all[k, y_buckets_all[k, i]] for k in range(n_hash)]

        # the median approach
        did_set = False
        if estimates is not None and len(estimates) >= i:
            sketch_median = np.median(sketch_estimates)
            sorted_estimates = np.sort(sketch_estimates)
            sketch_std = np.abs(sorted_estimates[1] - sorted_estimates[3]) 
          
            # accept the estimate if it's within x standard deviations
            # from the median
            if np.abs(estimates[i] - sketch_median) <= 3*sketch_std:
                y_est[i] = estimates[i]
                did_set = True

        if not did_set:
            y_est[i] = np.median(sketch_estimates)


        # #the mean approach
        # did_set = False
        # if estimates is not None and len(estimates) >= i:
        #     sketch_mean = np.mean(sketch_estimates)
        #     sketch_std = np.std(sketch_estimates)
          
        #     # accept the estimate if it's within x standard deviations
        #     # from the mean
        #     if np.abs(estimates[i] - sketch_mean) <= 1*sketch_std:
        #         y_est[i] = estimates[i]
        #         did_set = True

        # if not did_set:
        #     y_est[i] = np.median(sketch_estimates)


        ## the median approach
        # if estimates is not None and len(estimates) >= i:
        #     sketch_estimates.append(estimates[i])
        
        # y_est[i] = np.median(sketch_estimates)

        # if i < 10:
        #     print("(sketch) real: " + str(int(y[i])) + " est: " + str(int(y_est[i])) + " delta: " + str(int(np.abs(y[i] - y_est[i]))))
        #     #print("real prob: " + str(y_prob[i-start]) + " pred prob: " + str(scaling_factor*y_pred[i - start]))


        if loss_function != None:
            loss += loss_function(y[i], y_est[i])

    return y_est, loss/len(y)







import os
import sys
import time
import argparse
import random

import numpy as np
from sketches import count_min, count_sketch
from sketch_utils import random_hash_with_sign, order_y_wkey_list, random_hash_float32, random_hash_with_sign_float32

# computes observed - predicted for each value
# if negativecutoff=True then all negative values are 
# set to zero
def get_pred_diffs(y, y_scores, negativecutoff=True):
    y = y.astype('float32')
    y_scores = y_scores.astype('float32')

    # the predicted scores. recall: y_scores in *logarithmic* scale
    pred_scores = np.exp(y_scores)

    # sum of all predicted scores
    score_total = np.sum(pred_scores)

    # sum of all value frequencies
    y_total = np.sum(y)

    # the predicted counts for each item
    pred_counts = y_total * (pred_scores / score_total)
    
    # difference in the predicted and actual counts
    diffs = np.subtract(y, pred_counts).astype('float32')
   
    # set all negative values to zero
    diffs[diffs < 0.0] = 0.0
    diffs = diffs.astype('float32')

    return diffs, pred_counts

# learned_count_min_sketch uses the frequency prediction oracle to 
# store the difference between the predicted and observed counts.
# The sketch stores the (obs_count - pred_count) as a single precision 
# floating point value using np.float32 and outputs the prediction as
# sketch(value) + pred(value)
def learned_count_min_sketch(y, y_scores, n_hash, n_buckets, loss_function):
    if len(y) == 0:
        return 0    # avoid division of 0

    # compute observed - predicted counts
    diffs, pred_counts = get_pred_diffs(y, y_scores, negativecutoff=True)

    # counts returned by the sketching alg
    sketch_counts = np.zeros((n_hash, n_buckets)).astype('float32')

    # buckets in the sketching alg
    sketch_buckets_all = np.zeros((n_hash, len(y)), dtype=int)

    # sketch the diffs using count min sketch
    for i in range(n_hash):
        counts, loss, y_buckets = random_hash_float32(diffs, n_buckets)
        sketch_counts[i] = counts
        sketch_buckets_all[i] = y_buckets

    # compute the loss of the algorithm
    loss = 0
    for i in range(len(y)):
        y_est = np.min([sketch_counts[k, sketch_buckets_all[k, i]] for k in range(n_hash)])
        y_est += pred_counts[i] # add the predicted count back in

        # apply the loss function of the actual vs. estimated count
        loss += loss_function(y[i], y_est)
        
    return loss / len(y)

# learned_count_sketch uses the frequency prediction oracle to 
# store the difference between the predicted and observed counts.
# The sketch stores the (obs_count - pred_count) as a single precision 
# floating point value using np.float32 and outputs the prediction as
# sketch(value) + pred(value)
def learned_count_sketch(y, y_scores, n_hash, n_buckets, loss_function):
    if len(y) == 0:
        return 0    # avoid division of 0
    
    # compute observed - predicted counts
    diffs, pred_counts = get_pred_diffs(y, y_scores, negativecutoff=True)

    # counts returned by the sketching alg
    sketch_counts = np.zeros((n_hash, n_buckets)).astype('float32')

    # buckets in the sketching alg
    sketch_buckets = np.zeros((n_hash, len(y)), dtype=int)

    # signs in the sketching alg (see count sketch)
    sketch_signs = np.zeros((n_hash, len(y)), dtype=int)

    for i in range(n_hash):
        counts, buckets, signs = random_hash_with_sign_float32(diffs, n_buckets)
        sketch_counts[i] = counts
        sketch_buckets[i] = buckets
        sketch_signs[i] = signs

    loss = 0
    for i in range(len(y)):
        y_est = np.median(
            [sketch_signs[k, i] * sketch_counts[k, sketch_buckets[k, i]] for k in range(n_hash)])
        y_est += pred_counts[i] # add the predicted count back in
        y_est = np.abs(y_est)

        # apply the loss function of the actual vs. estimated count
        loss += loss_function(y[i], y_est)

    return loss / len(y)

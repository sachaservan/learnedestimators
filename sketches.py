import os
import sys
import time
import argparse
import random
import numpy as np

from utils import get_data
from collections import defaultdict
from sketch_utils import random_hash, random_hash_avg, random_hash_with_sign, random_hash_with_bucket_weights

def count_min(items, n_buckets, n_hash, loss_function=None):
    counts_all = np.zeros((n_hash, n_buckets))
    item_buckets_all = np.zeros((n_hash, len(items)), dtype=int)
    for i in range(n_hash):
        counts, loss, item_buckets = random_hash(items, n_buckets)
        counts_all[i] = counts
        item_buckets_all[i] = item_buckets

    loss = 0
    item_est = np.zeros(len(items))
    for i in range(len(items)): 
        sketch_estimates = [counts_all[k, item_buckets_all[k, i]] for k in range(n_hash)]
        item_est[i] = np.min(sketch_estimates)

        if loss_function != None:
            loss = loss_function(loss, items[i], item_est[i])

    return item_est, loss


def count_min_with_weights(items, n_buckets, n_hash, loss_function=None, weights=None):
    if len(items) == 0:
        return 0    # avoid division of 0

    counts_all = np.zeros((n_hash, n_buckets))
    item_buckets_all = np.zeros((n_hash, len(items)), dtype=int)
    for i in range(n_hash):
        counts, weights, item_buckets = random_hash_with_bucket_weights(items, n_buckets, weights=weights)
        counts_all[i] = counts
        item_buckets_all[i] = item_buckets

    loss = 0
    item_est = np.zeros(len(items))
    for i in range(len(items)): 
        sketch_estimates = [counts_all[k, item_buckets_all[k, i]] for k in range(n_hash)]
        item_est[i] = np.min(sketch_estimates)

        if loss_function != None:
            loss = loss_function(loss, items[i], item_est[i])

    return item_est, loss


# count sketch algorithm augmented with the ability to 
# to select from a given set of values provided if within 
# std standard deviations from the mean computed by countsketch
def count_sketch(items, n_buckets, n_hash, loss_function=None, estimates=None, sd_thresh=2):

    num_items = len(items)
    selections = np.zeros(num_items, dtype=int)

    counts_all = np.zeros((n_hash, n_buckets))
    item_buckets_all = np.zeros((n_hash, num_items), dtype=int)
    item_signs_all = np.zeros((n_hash, num_items), dtype=int)
    for i in range(n_hash):
        counts, item_buckets, item_signs = random_hash_with_sign(items, n_buckets)
        counts_all[i] = counts
        item_buckets_all[i] = item_buckets
        item_signs_all[i] = item_signs

    loss = 0
    item_est = np.zeros(num_items)
    for i in range(num_items):
        sketch_estimates = [item_signs_all[k, i] * counts_all[k, item_buckets_all[k, i]] for k in range(n_hash)]

        # the median approach
        did_set = False
        if estimates is not None and len(estimates) >= i:
            sketch_median = np.median(sketch_estimates)
            sorted_estimates = np.sort(sketch_estimates)
            sketch_sd = np.abs(sorted_estimates[1] - sorted_estimates[3]) 
          
            # accept the estimate if it's within x standard deviations
            # from the median
            if np.abs(estimates[i] - sketch_median) <= sd_thresh*sketch_sd:
                item_est[i] = estimates[i]
                did_set = True
                selections[i] = 1

        if not did_set:
            item_est[i] = np.median(sketch_estimates)


        if loss_function != None:
            loss = loss_function(loss, items[i], item_est[i])

    return item_est, loss, np.sum(selections) / len(items)







import os
import sys
import time
import argparse
import random
import numpy as np
import math

from utils import get_data
from collections import defaultdict
from sketch_utils import random_hash, random_hash_avg, random_hash_with_sign, random_hash_with_bucket_weights, random_hash_with_sign_and_weights


''' 
regular count min algorithm 
'''
def count_min(items, n_buckets, n_hash):
    counts_all = np.zeros((n_hash, n_buckets))
    item_buckets_all = np.zeros((n_hash, len(items)), dtype=int)
    for i in range(n_hash):
        counts, loss, item_buckets = random_hash(items, n_buckets)
        counts_all[i] = counts
        item_buckets_all[i] = item_buckets

    item_est = np.zeros(len(items))
    for i in range(len(items)): 
        sketch_estimates = [counts_all[k, item_buckets_all[k, i]] for k in range(n_hash)]
        item_est[i] = np.min(sketch_estimates)

    return item_est


''' 
regular count sketch algorithm 
'''
def count_sketch(y, n_buckets, n_hash):
    item_est = np.zeros(len(y))
    counts_all = np.zeros((n_hash, n_buckets))
   
    y_buckets_all = np.zeros((n_hash, len(y)), dtype=int)
    y_signs_all = np.zeros((n_hash, len(y)), dtype=int)
    
    for i in range(n_hash):
        counts, y_buckets, y_signs = random_hash_with_sign(y, n_buckets)
        counts_all[i] = counts
        y_buckets_all[i] = y_buckets
        y_signs_all[i] = y_signs

    for i in range(len(y)):
        item_est[i] = np.abs(np.median(
            [y_signs_all[k, i] * counts_all[k, y_buckets_all[k, i]] for k in range(n_hash)]))
    
    return item_est


''' 
count sketch algorithm augmented with the ability to 
to select from a given set of values provided if within 
std standard deviations from the mean computed by countsketch
'''
def count_sketch_with_estimates(items, n_buckets, n_hash,  estimates=None, sd_thresh=2):
    num_items = len(items)
    counts_all = np.zeros((n_hash, n_buckets))
    collisions_all = np.zeros((n_hash, n_buckets))
    item_buckets_all = np.zeros((n_hash, num_items), dtype=int)
    item_signs_all = np.zeros((n_hash, num_items), dtype=int)
    for i in range(n_hash):
        counts, collisions, item_buckets, item_signs = random_hash_with_sign_and_weights(items, n_buckets)
        counts_all[i] = counts
        item_buckets_all[i] = item_buckets
        item_signs_all[i] = item_signs
        collisions_all[i] = collisions

    sum_estimates = np.sum(estimates)

    item_est = np.zeros(num_items)
    for i in range(num_items):
        sketch_estimates = np.array([item_signs_all[k, i] * counts_all[k, item_buckets_all[k, i]] for k in range(n_hash)])
        sketch_median = np.abs(np.median(sketch_estimates))

        # the median approach
        did_set = False
        if estimates is not None and len(estimates) >= i:
            sketch_collisions = np.array([collisions_all[k, item_buckets_all[k, i]] for k in range(n_hash)])
            sketch_counts = np.array([counts_all[k, item_buckets_all[k, i]] for k in range(n_hash)])
            sketch_est = np.abs(np.median(sketch_estimates / sketch_collisions))
            sketch_sd = np.std(np.abs(sketch_counts) / sketch_collisions)

            # accept the estimate if it's within x standard deviations
            # from the mean
            if np.abs(sketch_est - estimates[i]) < sd_thresh * sketch_sd:
                did_set = True
                item_est[i] = estimates[i]

                # if i < 100:
                #     print("selected oracle instead of sketch. " )
                #     print("mean: " + str(sketch_mean) + " median: " + str(sketch_median) + " sd: " + str(sketch_sd))
                #     print("Oracle: " + str(estimates[i]))
                #     print("Sketch: " + str(sketch_median))
                #     print("True: " + str(items[i]))

        if not did_set:
            item_est[i] = sketch_median

    return item_est

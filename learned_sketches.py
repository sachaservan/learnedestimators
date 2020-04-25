import os
import sys
import time
import argparse
import random
import math
import numpy as np
from sketches import count_min, count_sketch
from weighted_distinct_elements import compute_sum_estimate


# learned_count_min_sketch uses the frequency prediction oracle 
# and count min sketch 
def learned_count_sketch_just_cutoff(items, scores, n_hash, n_buckets, loss_function):
    if len(items) == 0:
        return 0, 0    # avoid division of 0

    # cutoff the most frequent items
    items = items[n_buckets:]
    scores = scores[n_buckets:]
    n_buckets -= int(n_buckets / n_hash) # used 1*n_buckets worth of bytes to store top frequent items 

    # get count min sketch estimates for the remaining (non cutoff) items
    _, loss, _ = count_sketch(items, n_buckets, n_hash, loss_function=loss_function) 

    return loss

# learned_count_min_sketch uses the frequency prediction oracle 
# and count min sketch 
def learned_count_sketch(items, scores, n_hash, n_buckets, loss_function, cutoff=False):
    if len(items) == 0:
        return 0, 0    # avoid division of 0

    if cutoff:
        items = items[n_buckets:]
        scores = scores[n_buckets:]
        n_buckets -= int(n_buckets / n_hash) # used 1*n_buckets worth of bytes to store top frequent items 


    # fraction of space used relative to the total space needed to store entire stream
    # this is used to determine the threshold for choosing betweeen sketch vs. oracle predictions
    space_fraction = (n_buckets * n_hash) / len(items)

    # choose oracle if oracle pred < sd_threshold standard deviations from the sketch prediction
    sd_threshold = (1.0 - space_fraction) * 3 
   
    fix_point = 1000
    scores_fp = (scores * fix_point).astype(int) # fixpoint encode the real values
    scores_total, space_in_bytes = compute_sum_estimate(scores_fp, 0.05, n_reg=128)
    scores_total /= fix_point # scale back down to float

    print("estimated sum: " + str(scores_total))
    print("true sum:      " + str(np.sum(scores)))
    print("space used to compute sum: " + str(space_in_bytes) + " bytes")

    # reduce space for sketch that was allocated to the weighted element data structure
    n_buckets -= int(space_in_bytes / n_hash / 4) # each bucket is 4 bytes and there are n_buckets * n_hash * 4 bytes in total

    # conver the oracle predicted scores into a frequncy prediction
    freq_pred = scores / scores_total
    estimates = np.sum(items) * freq_pred  # compute estimated countrs for each item

    # get count min sketch estimates with hints
    _, loss, percent_oracle = count_sketch(items, n_buckets, n_hash, loss_function=loss_function, estimates=estimates, sd_thresh=sd_threshold) 

    return loss, percent_oracle

import os
import sys
import time
import argparse
import random

import numpy as np
from sketches import count_min, count_sketch
from sketch_utils import random_hash, random_hash_avg, random_hash_with_scores, random_hash_with_sign, order_y_wkey_list

# test code
def oracle_count_min_sketch_with_floats(y, y_scores, n_hash, n_buckets, loss_function):
    if len(y) == 0:
        return 0    # avoid division of 0
    
    counts_all = []
    y_buckets_all = []

    splits, idxmap = equisplit(y, y_scores, n_hashes)

    # reverse the splits to have largest size first
    splits_rev = splits[::-1]
    sizes = [len(splits_rev[k]) for k in range(n_hashes)]

    splitvalues = []
    for j in range(n_hashes):
        splitvalues.extend(splits[j])

        # calculate percentage of buckets allcated to this block
        percentage = sizes[j] / len(y)
        n_buckets_j = int(np.ceil(n_hashes * n_buckets * percentage))       

        counts, loss, y_buckets = random_hash(splitvalues, n_buckets_j)
        counts_all.append(counts)
        y_buckets_all.append(y_buckets.tolist())

    loss = 0
    for i in range(len(y)):
        j = idxmap[i]

        index = i
        counts = []
        for k in range(j, n_hashes):
            counts.append(counts_all[k][y_buckets_all[k][index]])
      
        y_est = np.min(np.array(counts))
        loss += loss_function(y[i], y_est)
        assert(y_est >= y[i]) 
   
    return loss / len(y)

# dynamic version of count sketch 
# more hashes are used the more frequent the item is
# n_hash_min: determines the minimum number of hash functions to use
# i.e., the most frequent elements will be hashed using n_hash_min hashes
# n_hash_max: number of functions for the least frequent elements
def dynamic_count_sketch(y, y_scores, n_hash, n_buckets, loss_function):
    if len(y) == 0:
        return 0  # avoid division of 0

    n_batches = 1000
    diffs = np.zeros(len(y))
    score_total = np.sum(np.exp(y_scores))
    y_batch = np.copy(y)
    y_batch = np.floor(y_batch / n_batches)
    y_batch.astype(int)
    y_total = int(np.sum(y_batch))
    pred_counts = np.floor(y_total * np.exp(y_scores) / score_total)

    for i in range(n_batches):
        diffs = np.add(diffs, np.subtract(y_batch, pred_counts))
    
    # set all negative values to zero
    diffs[diffs < 0] = 0 
        
    counts_all = []
    y_buckets_all = []
    y_signs_all = []

    # set the pred counts to be that of all batches
    pred_counts = np.floor(n_batches * y_total * np.exp(y_scores) / score_total)

    for i in range(n_hash):
        counts, y_buckets, y_signs = random_hash_with_sign(diffs, n_buckets)
        counts_all.append(counts)
        y_buckets_all.append(y_buckets.tolist())
        y_signs_all.append(y_signs)

    loss = 0
    for i in range(len(y)):
        counts = []
        for k in range(n_hash):
            counts.append(y_signs_all[k][i]*counts_all[k][y_buckets_all[k][i]])

        y_est = np.median(np.array(counts))
        y_est += pred_counts[i]
        loss += loss_function(y[i], y_est)  
        
    return loss / len(y)

    

# dynamic version of count min 
# more hashes are used the more frequent the item is
# n_hash_min: determines the minimum number of hash functions to use
# i.e., the most frequent elements will be hashed using n_hash_min hashes
# n_hash_max: number of functions for the least frequent elements
def dynamic_count_min(y, y_scores, n_hashes, n_buckets, loss_function):
    if len(y) == 0:
        return 0    # avoid division of 0

    counts_all = []
    y_buckets_all = []

    splits, idxmap = equisplit(y, y_scores, n_hashes)

    # reverse the splits to have largest size first
    splits_rev = splits[::-1]
    sizes = [len(splits_rev[k]) for k in range(n_hashes)]

    splitvalues = []
    for j in range(n_hashes):
        splitvalues.extend(splits[j])

        # calculate percentage of buckets allcated to this block
        percentage = sizes[j] / len(y)
        n_buckets_j = int(np.ceil(n_hashes * n_buckets * percentage))       

        counts, loss, y_buckets = random_hash(splitvalues, n_buckets_j)
        counts_all.append(counts)
        y_buckets_all.append(y_buckets.tolist())

    loss = 0
    for i in range(len(y)):
        j = idxmap[i]

        index = i
        counts = []
        for k in range(j, n_hashes):
            counts.append(counts_all[k][y_buckets_all[k][index]])
      
        y_est = np.min(np.array(counts))
        loss += loss_function(y[i], y_est)
        assert(y_est >= y[i]) 
   
    return loss / len(y)


def equisplit(values, probs, n_chuncks):
    idxmap = {}
    splits = []
    target = int(np.floor(np.sum(probs)/float(n_chuncks)))
    i = 0
    psum = 0
    csplit =[]
    for k in range(len(values)):
        psum += probs[k]
        idxmap[k] = i
        csplit.append(values[k])
        if psum > target:
            splits.append(csplit)
            csplit = []
            psum = 0
            i  += 1
    
    if len(csplit) >= 0:
        splits.append(csplit)

    return splits, idxmap
        
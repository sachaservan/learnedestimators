import os
import sys
import time
import argparse
import random
import math
import numpy as np
from sketches import count_min, count_sketch
from sketch_utils import random_hash_with_sign, order_y_wkey_list, random_hash_with_sign, random_hash, random_hash_with_scores

ORDERING_RANDOM = 0
ORDERING_HIGH_LOW = 1
ORDERING_LOW_HIGH = 2

# returns a mask where `True` for all elements that had false-positives
# ording determines what order the elements arrive into the bloom filter
def bloom_filter(y_original, n_reg, n_hash, ordering):

    y = y_original.copy() # copy over the list
    if ordering == ORDERING_HIGH_LOW:
       y = np.sort(y)[::-1]
    elif ordering == ORDERING_RANDOM:
       np.random.shuffle(y)
    elif ordering == ORDERING_LOW_HIGH:
       y = np.sort(y)

    y_false_positives = np.zeros(len(y), dtype=bool)
    y_reg_all = np.zeros(n_hash*n_reg, dtype=bool)
    hashes = np.random.choice(np.arange(n_reg*n_hash), size=len(y)*n_hash)
    for i in range(len(y)): 
        n_hits = 0
        for j in range(n_hash):
            h_i = hashes[i*n_hash + j]
            if y_reg_all[h_i]:
                n_hits += 1
            y_reg_all[h_i] = True # mask all indices with True

        if n_hash >= n_hits:
            y_false_positives[i] = True # mark as false-positive

    return y_false_positives

def hyperloglogsupports(buckets, n_reg):
    supps = np.zeros(len(buckets))

    assert(n_reg >= 128) # if < 128, need different correction value

    for i in range(len(buckets)): 
        # compute hyperloglog
        m = n_reg # number of registers
        lm = int(math.log(m, 2))

        assert(lm < 64) # the "hash" assumes 128 bits of which the first lm are used

        a = 0.7213 / (1 + 1.079/m) # correction value

        registers = np.zeros(m, dtype=int)
        for j in range(buckets[i]):
            bval = np.random.choice(np.arange(2), size=m)
            regidx = 0
            for k in range(0, lm): # compute register index
                regidx += (1 << k) * bval[k]

            msbidx = 0 # index (+1) of most significant bit in 
            for k in range(len(bval[lm:])):
                if bval[lm:][k] == 1:
                    msbidx = k
                    break

            if msbidx > registers[regidx]:
                registers[regidx] = msbidx
        
        # hyperloglog estimate
        z = 0.0
        for k in range(len(registers)): 
            s = (1 << registers[k])
            z += 1.0/s

        supps[i]  = a * (m**2) / z

    return supps

# splits the values such that 
# each split contains an equal probability mass
def equisplit(values, scores, n_chuncks):
    moment = 1
    splits = np.zeros(n_chuncks, dtype=int)
    split_scores = np.zeros(n_chuncks)
    target = np.sum(scores**moment)/n_chuncks
    i = 0
    scoresum = 0
    count = 0
    for k in range(len(values)):
        scoresum += scores[k]**moment
        count += 1
        if scoresum >= target:
            splits[i] = count
            split_scores[i] = scoresum / count
            count = 0
            scoresum = 0
            i += 1

    splits[i] = count
    split_scores[i] = scoresum / count

    return splits 


def second_moment_estimate(y):
    iters = 5
    est = 0
    for i in range(iters):
        y_signs = np.random.choice([-1, 1], size=len(y))
        est += (np.inner(y_signs, y))**2 # inner product squared

    return est / iters


# learned_count_min_sketch uses the frequency prediction oracle 
# and count min sketch 
def learned_count_sketch(y, y_scores, n_hash, n_buckets, loss_function, cutoff=False):
    if len(y) == 0:
        return [], 0    # avoid division of 0

    if cutoff:
        y = y[n_buckets:]
        y_scores = y_scores[n_buckets:]
        n_hash -= 1 # used 1*n_buckets worth of bytes

    bloom_filter_bytes = 100 # using 8 hashes for total space of 100*8 bits
    bloom_mask = bloom_filter(y, bloom_filter_bytes , 8, ordering=ORDERING_RANDOM)
   
    n_buckets -= int(bloom_filter_bytes/n_hash) # reduce space for sketch that was allocated to the bloom filter

    y_scores_filtered = y_scores.copy()
    y_scores_total = np.sum(y_scores_filtered)    

    y_pred = y_scores / y_scores_total
    estimates = np.sum(y) * y_pred

    # get count min sketch estimates with hints
    _, loss = count_sketch(y, n_buckets, n_hash, loss_function=loss_function, estimates=estimates) 

    return loss


# learned_count_min_sketch uses the frequency prediction oracle 
# and count min sketch 
def learned_count_min_sketch(y, y_scores, n_hash, n_buckets, loss_function, cutoff=False):
    if len(y) == 0:
        return [], 0    # avoid division of 0

    if cutoff:
        y = y[n_buckets:]
        y_scores = y_scores[n_buckets:]
        n_hash -= 1

    total_space = n_buckets

    # parameters 
    bucket_partitions = 10
    n_buckets = int(total_space / bucket_partitions)


    y_scores_total = np.sum(y_scores)  
    y_pred = y_scores / y_scores_total

    # split using the "hints" provided by oracle
    buckets = equisplit(y, y_pred, bucket_partitions)

    loss = 0
    curr_b = 0 # current bucket
    start = 0 # start index in y for bucket i
    end = buckets[curr_b]

    bucket_proportion = (len(y) - (end - start)) / len(y)
    n_hash_bucket = max(int(bucket_partitions * bucket_proportion), 1)

    y_est_bucket = []
    if end - start < n_buckets:
        y_est_bucket = y[start:end]
    else:
        y_est_bucket, _ = count_min(y[start:end], n_buckets, n_hash_bucket)

    # begin computing loss
    for i in range(len(y)):
        # ith element belongs to the next bucket
        if i - start >= buckets[curr_b]:
            curr_b += 1
            start = i
            end = start + buckets[curr_b]

            bucket_proportion = (len(y) - (end - start)) / len(y)
            n_hash_bucket = max(int(bucket_partitions * bucket_proportion), 1)
            n_buckets_bucket = n_buckets 


            if end - start < n_buckets:
                y_est_bucket = y[start:end]
            else:
                y_est_bucket, _ = count_min(y[start:end], n_buckets, n_hash_bucket)

        loss += loss_function(y[i], y_est_bucket[i-start])

        if i < 100:
            print("(learned) real: " + str(int(y[i])) + " est: " + str(int(y_est_bucket[i-start])) + " delta: " + str(int(np.abs(y[i] - y_est_bucket[i-start]))))


    if np.sum(buckets) != len(y):
        print("Error: something is wrong...lengths don't match")

    return loss / len(y)



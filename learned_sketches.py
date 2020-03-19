import os
import sys
import time
import argparse
import random
import math
import numpy as np
from sketches import count_min, count_sketch, count_mean
from sketch_utils import random_hash_with_sign, order_y_wkey_list, random_hash_with_sign, random_hash

# splits the values such that 
# each split contains an equal probability mass
def equisplit(values, scores, n_chuncks):
    moment = 4  #TODO(sss) figure this value out
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
            bval = np.random.choice(np.arange(2), size=64)
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


# learned_count_min_sketch uses the frequency prediction oracle 
# and count min sketch 
def learned_constant_predictor(y, y_scores, n_hash, n_buckets, loss_function):
    if len(y) == 0:
        return [], 0    # avoid division of 0

    n_buckets = int(math.sqrt(n_buckets*n_hash)) 

    # hyperloglog parameters
    n_reg = 2**int(math.log(n_buckets, 2)) # 2^(sqrt(B))
    
    # hash using the probabiliy
    buckets = equisplit(y, y_scores, n_buckets)

    # estimate support of each bucket using HLL
    est_supps = hyperloglogsupports(buckets, n_reg)
    
    # HLL space: 
    # each register is 4 bytes (i.e., one bucket) have 2^sqrt(B) registers
    loss = 0
    curr_b = 0 # current bucket
    start = 0 # start index in y for bucket i
    end = buckets[curr_b]
    
    total_in_bucket_i = np.sum(y[start: end])

    # begin computing loss
    for i in range(len(y)):

        # ith element belongs to the next bucket
        if i - start >= buckets[curr_b]:
            curr_b += 1
            start = np.sum(buckets[:curr_b])
            end = start + buckets[curr_b]
            total_in_bucket_i = np.sum(y[start: end])

        # perform estimate 
        y_est = total_in_bucket_i / est_supps[curr_b]

        if i < 100:
            print("real: " + str(y[i]) + " est: " + str(y_est) + " delta: " + str(int(np.abs(y[i] - y_est))))

        loss += loss_function(y[i], y_est)

    if np.sum(buckets) != len(y):
        print("Error: something is wrong...lengths don't match")

    return loss / len(y)

# learned_count_min_sketch uses the frequency prediction oracle 
# and count min sketch 
def learned_scaling_predictor(y, y_scores, n_hash, n_buckets, loss_function):
    if len(y) == 0:
        return [], 0    # avoid division of 0
    
    # split using the "hints" provided by oracle
    buckets = equisplit(y, y_scores, n_buckets*n_hash)

    loss = 0
    curr_b = 0 # current bucket
    start = 0 # start index in y for bucket i
    end = buckets[curr_b]

    y_pred = np.exp(y_scores[start:end]) / np.sum(np.exp(y_scores[start:end]))
    y_prob = y[start:end] / np.sum(y[start:end])
   
    y_est_bucket = np.inner(y_prob, y_pred) # inner product between "hints" and actual observed counts
    second_moment_est = np.sum(y_pred**2) # HACK: computing it exactly here...
    scaling_factor = y_est_bucket / second_moment_est
    total_in_bucket_i = np.sum(y[start:end])

    print("actual prob " + str(y_prob[:100]))
    print("pred prob " + str(y_pred[:100]))
    print("scaling factor " + str(scaling_factor))
    
    # begin computing loss
    for i in range(len(y)):
        # ith element belongs to the next bucket
        if i - start >= buckets[curr_b]:
            curr_b += 1
            start = np.sum(buckets[:curr_b])
            end = start + buckets[curr_b]

            y_pred = np.exp(y_scores[start:end]) / np.sum(np.exp(y_scores[start:end]))
            y_prob = y[start:end] / np.sum(y[start:end])
            y_est_bucket = np.inner(y_prob, y_pred) # inner product between "hints" and actual observed counts
            second_moment_est = np.sum(y_pred**2) # HACK: computing it exactly here...
            scaling_factor = y_est_bucket / second_moment_est
            total_in_bucket_i = np.sum(y[start:end])


        # compute scaling estimate 
        y_est = scaling_factor*y_pred[i - start]*total_in_bucket_i

        if i < 100:
            print("real: " + str(y[i]) + " est: " + str(y_est) + " delta: " + str(np.abs(y[i] - y_est)))
            #print("real prob: " + str(y_prob[i-start]) + " pred prob: " + str(scaling_factor*y_pred[i - start]))

        loss += loss_function(y[i], y_est)

    if np.sum(buckets) != len(y):
        print("Error: something is wrong...lengths don't match")

    return loss / len(y)


# learned_count_min_sketch uses the frequency prediction oracle 
# and count min sketch 
def learned_hybrid(y, y_scores, n_hash, n_buckets, loss_function):
    if len(y) == 0:
        return [], 0    # avoid division of 0

    n_buckets = max(int(math.sqrt(n_buckets)), 128) # hack in case < 128

    # hyperloglog parameters
    n_reg = 2**int(math.log(n_buckets, 2)) # 2^(sqrt(B))
    
    # hash using the probabiliy
    buckets = equisplit(y, y_scores, n_buckets)

    # estimate support of each bucket using HLL
    est_supps = hyperloglogsupports(buckets, n_reg) 
    # HLL space: 
    # each register is 4 bytes (i.e., one bucket) have 2^sqrt(B) registers

    n_hash -= 1

    loss = 0
    curr_b = 0 # current bucket
    start = 0 # start index in y for bucket i
    is_sketch = True # whether or not the

    # compute count min estimates for the first bucket
    end = buckets[curr_b]
    y_est_bucket = []
        
    if len(y[start:end]) >= n_buckets*n_hash:
            y_est_bucket, _ = count_min(y[start:end], n_buckets, n_hash) # get count min sketch estimates
    else:
        y_est_bucket = y[start:end] 
        is_sketch = False

    # begin computing loss
    for i in range(len(y)):
        # ith element belongs to the next bucket
        if i - start >= buckets[curr_b]:
            curr_b += 1
            start = np.sum(buckets[:curr_b])
            end = start + buckets[curr_b]
            
            if len(y[start:end]) >= n_buckets*n_hash:
                is_sketch = True
                y_est_bucket, _ = count_min(y[start:end], n_buckets, n_hash) # get count min sketch estimates
            else:
                y_est_bucket = y[start:end] 
                is_sketch = False

        # perform estimate 
        y_est = y_est_bucket[i-start]
        if is_sketch:
            expected_collisions = est_supps[curr_b] / n_buckets 
            y_est = y_est / expected_collisions

        if i < 10:
            print("real: " + str(y[i]) + " est: " + str(y_est) + " delta: " + str(np.abs(y[i] - y_est)))

        loss += loss_function(y[i], y_est)


    if np.sum(buckets) != len(y):
        print("Error: something is wrong...lengths don't match")

    return loss / len(y)

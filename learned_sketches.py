import os
import sys
import time
import argparse
import random
import math
import numpy as np
from sketches import count_min, count_sketch
from weighted_distinct_elements import compute_sum_estimate
from sketch_utils import random_hash_with_sign_and_weights
from weighted_distinct_elements import hyperloglogsimulate

# TODO: cmake this a parameter
SPACE_IN_BYTES_FOR_WIGHTED_DISTINCT_ELEMENTS = 2000


'''
weighted splits returns n arrays with the inputs probs split
such that 1) there are at least min_size elements per buckets 
and 2) the min-max arange in each bucket is smallest
(implemented as a greedy algorithm) 
'''
def weighted_split(probs, n, min_size=1, eps=0.05):
    splits = np.array_split(probs, n)
    
    didChange = True
    
    dec = int(eps*len(probs) / n)
    fracs = [len(x)-int(eps*len(x)) for x in splits]
    while didChange:

        didChange = False
        for i in range(n-1):
            
            delta_left = np.std(splits[i])
            delta_right = np.std(splits[i+1])

            concat = np.concatenate((splits[i][fracs[i]:], splits[i+1]), axis=0)
            delta_right_add =  np.std(concat)
            delta_left_remove = np.std(splits[i][:fracs[i]])

            if fracs[i] <= min_size or len(splits[i]) <= min_size:
                continue
         
            if delta_left_remove >= delta_right_add:
                splits[i+1] = concat
                splits[i] = splits[i][:fracs[i]]
                fracs[i] = len(splits[i]) - int(eps*len(splits[i]))
                fracs[i+1] = len(splits[i+1]) - int(eps*len(splits[i+1]))
                didChange = True
            
            # print(str(delta_left_remove) + " > " +  str(delta_right_add) + " minbucket = " + str(min_size)) 
            # lens = [len(x) for x in splits]
            # stds = [np.abs(np.mean(x) - np.median(x)) for x in splits]
            # print()
            # print(lens)
            # print(stds)
            # print()

    lens = [len(x) for x in splits]
    stds = [np.abs(np.mean(x) - np.median(x)) for x in splits]
    print()
    print(lens)
    print(stds)
    print()

    return splits

# learned_count_min_sketch uses the frequency prediction oracle 
# and count min sketch 
def learned_count_sketch_just_cutoff(items, scores, n_hash, n_buckets):
    
    # resulting estimates returned for each item 
    item_est = np.zeros(len(items))

    # TODO: figure out what cutoff to use
    cutoff_thresh = int(min(n_buckets, len(items) * 0.1)) # store top 10% of items in table or n_buckets if smaller
   
    # store perfect predictions for all cutoff items
    for i in range(cutoff_thresh):
        item_est[i] = items[i] 

    # cutoff the most frequent items
    items = items[cutoff_thresh:]
    n_buckets -= int(cutoff_thresh*2/n_hash) # need extra 4 bytes to store ID of element; hence mult by 2

    # get count min sketch estimates for the remaining (non cutoff) items
    estimates = count_sketch(items, n_buckets, n_hash) 

    for i in range(len(estimates)):
        item_est[cutoff_thresh + i] = estimates[i]

    return item_est

# learned_count_min_sketch uses the frequency prediction oracle 
# and count min sketch 
def learned_count_sketch_partitions(items, estimates, n_hash, n_buckets, cutoff=False):

    space = n_hash * n_buckets
    n_hash = 2 # we're using count-min; use 2 hash functions
    n_buckets = int(space / n_hash)

    # resulting estimates returned for each item 
    item_est = np.zeros(len(items))
    cutoff_thresh = 0
    if cutoff:
        # need extra 4 bytes to store ID of element;
        cutoff_thresh = int(min(n_buckets, len(items) * 0.1)) # need extra 4 bytes to store ID of element; hence mult by 2
        print("cutoff threshold is " + str(cutoff_thresh))
   
        # store perfect predictions for all cutoff items
        for i in range(cutoff_thresh):
            item_est[i] = items[i] 

        # cutoff the most frequent items
        items = items[cutoff_thresh:]
        estimates = estimates[cutoff_thresh:]

    # SANITY CHECKING
    n_buckets_original = n_buckets
    ####################################

    # TODO: figure out what the optimal number of partitions is...
    n_partitions = int(max(n_buckets / 4000, 10))
    n_count_sketch_partitions = -1 # number of partitions running count sketch

    # reduce space for sketch that was allocated to the weighted element data structure
    # each bucket is 4 bytes and there are n_buckets * n_hash * 4 bytes in total
    n_buckets -= int(SPACE_IN_BYTES_FOR_WIGHTED_DISTINCT_ELEMENTS / n_hash / 4) 

    # divvy up the number of buckets per partition 
    n_buckets = int(n_buckets / n_partitions) # number of buckets per partition 
    n_buckets = int(n_buckets / 1.25) # cost of keeping one byte counter per bucket (each bucket is 4bytes)

    # divide the estimates according to frequency predictions returned by the oracle
    # set the min size of a partition to be n_buckets/n_hash 
    splits = weighted_split(estimates, n_partitions, min_size=int(n_buckets/n_hash))
    sizes = [len(splits[k]) for k in range(n_partitions)]


    print("Partition sizes: " + str(sizes))
    print("Partition buckets: " + str(n_buckets))
    
    start = 0 # partition start index in the items list
    end = 0 # partition end index
    
    # SANITY CHECKING
    buckets_total_sanity_check = 0
    loss_sanity_check = 0
    loss_per_partition_sanity_check = 0
    ####################################

    # iterate over each partition and compute count-sketch or count-min with correction 
    for i in range(n_partitions):
        start = end
        end = start + sizes[i]
        part_items = items[start:end+1]

        n_buckets_partition = n_buckets # for now, each partition gets n_buckets

        # bookkeeping 
        # sum_part_items = np.sum(part_items)
        # part_mean = sum_part_items / len(part_items)
        # sum_part_estimates = np.sum(estimates[start:end+1])

        # for testing purposes
        loss_sanity_check += loss_per_partition_sanity_check
        loss_per_partition_sanity_check = 0
        buckets_total_sanity_check += n_buckets_partition

        # print("////////////////////////////////////")
        # print("partition size:    " + str(len(part_items)))
        # print("partition average: " + str(np.mean(part_items)))
        # print("estimates average: " + str(np.mean(estimates[start:end])))
        # print("estimates sd:      " + str(np.std(estimates[start:end])))
        # print("n_buckets:         " + str(n_buckets_partition))
        # print("////////////////////////////////////")

        counts_all = np.zeros((n_hash, n_buckets_partition))
        y_buckets_all = np.zeros((n_hash, len(part_items)), dtype=int)
        y_signs_all = np.zeros((n_hash, len(part_items)), dtype=int)
        collisions_all = np.zeros((n_hash, n_buckets_partition), dtype=int)
        for j in range(n_hash):
            count_min = True
            if i <= n_count_sketch_partitions:
                count_min = False

            # count sketch / count min with tracking of collisions per bucket
            # TODO: actually use Bloom filter to keep track of collisions 
            counts_all[j], collisions_all[j], y_buckets_all[j], y_signs_all[j] = random_hash_with_sign_and_weights(
                part_items, n_buckets_partition, countmin=count_min)
 
        # compute estimates for this partition
        for j in range(len(part_items)): 
            count_min = True
            if i <= n_count_sketch_partitions:
                count_min = False

            # counts per hash function 
            sketch_estimates = np.array([y_signs_all[k, j] * counts_all[k, y_buckets_all[k, j]] for k in range(n_hash)])

            # collisions per bucket / hash function 
            sketch_collisions = np.array([y_signs_all[k, j] * collisions_all[k, y_buckets_all[k, j]] for k in range(n_hash)])
        
            # sort = np.argsort(sketch_estimates)
            # sketch_collisions = sketch_collisions[sort]
            # sketch_estimates = sketch_estimates[sort]
            # second_moments = second_moments[sort]/n_hash

            if count_min:
                sort = np.argsort(sketch_estimates)
                sketch_collisions = sketch_collisions[sort]
                sketch_estimates = sketch_estimates[sort]
                
                sketch_estimates_corrected = sketch_estimates / sketch_collisions
                sketch_min = sketch_estimates_corrected[0]
                item_est[cutoff_thresh + start+j] = round(sketch_min)
                loss_per_partition_sanity_check += np.abs(item_est[cutoff_thresh+start+j] - part_items[j])
         
                if j < 0:
                    print()
                    print("partition size       " + str(len(splits[i])))
                    print("sketch_collisions    " + str(sketch_collisions))
                    print("estimates            " + str(sketch_estimates))
                    print("min (uncorrected)    " + str(np.min(sketch_estimates)))
                    print("min (corrected)      " + str(item_est[start+j]))
                    print("actual count:         " + str(part_items[j]))                 

            else:
                item_est[cutoff+start+j] = np.abs(np.median(sketch_estimates))
                loss_per_partition_sanity_check += np.abs(item_est[cutoff_thresh+start+j] - part_items[j])

            # if j < 5:
            #     print("--------------------------------" )
            #     print("Partition: " + str(i))
            #     print("Est:       " + str(item_est[start+j]))
            #     print("True:      " + str(items[start+j]))
            #     print("--------------------------------" )

        # standard deviations for each hash function
        # print("===================================")
        # print("max " + str(np.max(part_items)))
        # print("min " + str(np.min(part_items)))
        # print("std " + str(np.std(part_items)))
        # print("loss " + str(loss_per_partition_sanity_check / len(part_items)))
        # print("===================================")

    # make sure we're not using more buckets than originally allocated to the algorithm
    if buckets_total_sanity_check > n_buckets_original:
        print("ERROR: too many buckets")

    
    print("Total buckets used " + str(buckets_total_sanity_check))
    print("Total loss " + str(loss_sanity_check))
   
    return item_est


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
from scipy import stats
import matplotlib.pyplot as plt

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
    # print()
    # print(lens)
    # print(stds)
    # print()

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
def learned_count_sketch_partitions(items, scores, n_hash, n_buckets, cutoff=False):
   # SANITY CHECKING
    ####################################
    space_original = n_buckets*n_hash
    ####################################

    ####################################
    ########## BEGIN PARAMETERS 
    space = n_hash * n_buckets
    space_cs = space * 0.9
    space_cmin = space * 0.1 * 2

    # TODO: figure out what the optimal number of partitions is...
    n_partitions = 100
    n_count_sketch_partitions = 10 # number of partitions running count sketch

    # count sketch setup
    n_hash_partition_cs = 5
    n_buckets_partition_cs = int(space_cs / n_hash_partition_cs)

    # count min setup 
    n_hash_partition_cmin = 2
    n_buckets_partition_cmin = int(space_cmin / n_hash_partition_cmin / (n_partitions - n_count_sketch_partitions))
    n_buckets_partition_cmin = int(n_buckets_partition_cmin/1.25) # one byte counters needed for keeping track of collisions 

    ########## END PARAMETERS 

    ####################################
    ######### BEGIN CUTOFF 
    # resulting estimates returned for each item 
    item_est = np.zeros(len(items))
    cutoff_thresh = 0
    if cutoff:
        # need extra 4 bytes to store ID of element;
        cutoff_thresh = int(min(n_buckets_partition_cs, len(items) * 0.1)) # need extra 4 bytes to store ID of element; hence mult by 2
        print("cutoff threshold is " + str(cutoff_thresh))
   
        # store perfect predictions for all cutoff items
        for i in range(cutoff_thresh):
            item_est[i] = items[i] 

        # cutoff the most frequent items
        items = items[cutoff_thresh:]
        scores = scores[cutoff_thresh:]
    ######### END CUTOFF 

    
    # divide the estimates according to frequency predictions returned by the oracle
    # set the min size of a partition to be n_buckets/n_hash 
    splits = weighted_split(scores, n_partitions, min_size=int(n_buckets_partition_cmin))
    split_sizes = [len(splits[k]) for k in range(n_hash_partition_cs, n_partitions)]
    sizes = np.concatenate(([np.sum(split_sizes[:n_hash_partition_cs])], split_sizes[n_hash_partition_cs:]))
    n_partitions -= n_count_sketch_partitions

    print("Partition sizes:         " + str(sizes))
    print("Partition buckets CS:    " + str(n_buckets_partition_cs))
    print("Partition buckets CMIN:  " + str(n_buckets_partition_cmin))

    start = 0 # partition start index in the items list
    end = 0 # partition end index
    
    ####################################
    # SANITY CHECKING
    space_total_sanity_check = 0
    loss_sanity_check = 0
    loss_per_partition_sanity_check = 0
    number_of_items_processed_sanity_check = 0
    sum_indices_sanity_check = 0
    loss_per_partition = np.zeros(n_partitions)
    ####################################

    # iterate over each partition and compute count-sketch or count-min with correction 
    for i in range(n_partitions):
        start = end
        end = start + sizes[i]
        part_items = items[start:end]
        part_scores = scores[start:end]

        if i < n_count_sketch_partitions:
            count_min = False
            n_buckets_partition = n_buckets_partition_cs
            n_hash_partition = n_hash_partition_cs
            space_total_sanity_check += n_buckets_partition*n_hash_partition
        else:
            count_min = True
            n_buckets_partition = n_buckets_partition_cmin
            n_hash_partition = n_hash_partition_cmin
            space_total_sanity_check += n_buckets_partition*n_hash_partition / 2

        # bookkeeping 
        # sum_part_items = np.sum(part_items)
        part_mean = np.mean(part_items)
        # sum_part_estimates = np.sum(estimates[start:end+1])
        part_std = np.std(part_items)
        part_sum = np.sum(part_items)


        # compute score totals
        # TODO: use the weighted distinct element algorithm
        part_predictions = (part_scores * part_sum) / np.sum(part_scores)


        # for testing purposes
        loss_sanity_check += loss_per_partition_sanity_check
        loss_per_partition_sanity_check = 0

        counts_all = np.zeros((n_hash_partition, n_buckets_partition))
        y_buckets_all = np.zeros((n_hash_partition, len(part_items)), dtype=int)
        y_signs_all = np.zeros((n_hash_partition, len(part_items)), dtype=int)
        collisions_all = np.zeros((n_hash_partition, n_buckets_partition), dtype=int)

        for j in range(n_hash_partition):
            # count sketch / count min with tracking of collisions per bucket
            # TODO: actually use Bloom filter to keep track of collisions 
            counts, collisions, buckets, signs, = random_hash_with_sign_and_weights(
                part_items, n_buckets_partition, countmin=count_min)
            counts_all[j] = counts
            collisions_all[j] = collisions 
            y_buckets_all[j] = buckets
            y_signs_all[j] = signs 

        # compute estimates for this partition
        for j in range(len(part_items)): 
            # counts per hash function 
            sketch_estimates = np.array([y_signs_all[k, j] * counts_all[k, y_buckets_all[k, j]] for k in range(n_hash_partition)])
          
            # collisions per bucket / hash function 
            sketch_collisions = np.array([collisions_all[k, y_buckets_all[k, j]] - 1  for k in range(n_hash_partition)])
        
            # sort everything according to frequency counts 
            sort = np.argsort(sketch_estimates)
            sketch_estimates = sketch_estimates[sort]
            sketch_collisions = sketch_collisions[sort]
            # second_moments = second_moments[sort]/n_hash
        
            if count_min:
                # correct for bias         
                sketch_est = sketch_estimates[0] - sketch_collisions[0]*part_mean
                if sketch_est - part_mean >= part_std:
                    item_est[cutoff_thresh+start+j] = sketch_est
                else:
                    item_est[cutoff_thresh+start+j] = part_predictions[j]
            else:
                item_est[cutoff_thresh+start+j] = np.abs(np.median(sketch_estimates))

            # if np.abs(item_est[cutoff_thresh+start+j] - part_items[j]) > 100:         
            #     print()
            #     print("partition size       " + str(len(splits[i])))
            #     print("partition mode       " + str(part_mode))
            #     print("partition std        " + str(part_std))
            #     print("partition mean       " + str(part_mean))
            #     print("sketch_collisions    " + str(sketch_collisions))
            #     print("estimates            " + str(sketch_estimates))
            #     print("min (uncorrected)    " + str(np.min(sketch_estimates)))
            #     print("min (corrected)      " + str(item_est[start+j]))
            #     print("actual count:        " + str(part_items[j]))                 

            # make sure the loss per partition is reasonable
            number_of_items_processed_sanity_check += 1
            loss_per_partition_sanity_check += np.abs(item_est[cutoff_thresh+start+j] - part_items[j])
            sum_indices_sanity_check += cutoff_thresh+start+j

        loss_sanity_check += loss_per_partition_sanity_check
        loss_per_partition[i] = loss_per_partition_sanity_check
        print("////////////////////////////////////")
        print("partition size:    " + str(len(part_items)))
        print("partition average: " + str(part_mean))
        print("partition std:     " + str(part_std))
        print("n_buckets:         " + str(n_buckets_partition))
        print("loss:              " + str(loss_per_partition_sanity_check))
        print("////////////////////////////////////")

    # make sure we're not using more buckets than originally allocated to the algorithm
    if space_total_sanity_check > space_original:
        print("WARNING: too much used space; are all the parameters correct?")
    
    print("Total space used:  " + str(space_total_sanity_check))
    print("Total loss:        " + str(loss_sanity_check))
    print("# of partitions:   " + str(len(sizes)))
    print("# items processed: " + str(number_of_items_processed_sanity_check)  + " (" + str(len(items)) + " total items)")

    return item_est, loss_per_partition


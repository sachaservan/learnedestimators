import os
import sys
import time
import argparse
import random
import math
import numpy as np
from sketches import count_min, count_sketch
from sketch_common import hyperloglogsimulate, second_moment_estimate, random_hash, random_hash_with_sign

# constants used in experiments 
from experiment_constants import *

# setup logging 
import logging
logger = logging.getLogger('learned_estimators_log')

def estimate_median(counts_all, n_hash, n_collisions):
    est_median = 0
    for i in range(n_hash):
        est_median += np.median(counts_all[i]) / n_collisions  # estimate for the median in the hash buckets

    return max(1, est_median / n_hash)

def learned_count_sketch_threshold(items, scores, space, t=2):
    ''' 
    1) run regular CS and keep weighted distinct element count 
    2) if normalized oracle prediction is within t*SD from the mean value output normalized prediction; o/w output as CS
    '''

    # compute parameters to CS based on space 
    n_hash = COUNT_SKETCH_OPTIMAL_N_HASH 
    n_buckets = int(space / n_hash)
    n_buckets -= int(EXTRA_SPACE_PER_PARTITION_IN_BYTES / (n_hash * 4)) # space is in units of 4 bytes 

    ######################################################
    # 1) run vanilla CS 
    ######################################################
    cs_item_est = np.zeros(len(items)) # CS estimate for item count 
    cs_item_std = np.zeros(len(items)) # variance between buckets for each item 
   
    counts_all = np.zeros((n_hash, n_buckets))
    buckets_all = np.zeros((n_hash, len(items)), dtype=int)
    signs_all = np.zeros((n_hash, len(items)), dtype=int)
   
    for i in range(n_hash):
        counts, y_buckets, y_signs = random_hash_with_sign(items, n_buckets)
        counts_all[i] = counts
        buckets_all[i] = y_buckets
        signs_all[i] = y_signs
   
    for i in range(len(items)):
        sketch_ests = [signs_all[k, i] * counts_all[k, buckets_all[k, i]] for k in range(n_hash)]
        cs_item_est[i] = np.median(sketch_ests)
        cs_item_std[i] = np.std(sketch_ests)
    
    ######################################################
    # 2) do prediction selection 
    ######################################################

    # uses N_REGISTERS_FOR_HLL bytes (each regiseter is 1 byte) of memory to store distinct elements 
    n_distinct_elements = max(1, hyperloglogsimulate(len(items), N_REGISTERS_FOR_HLL)) # take max to avoid division by zero
    n_collisions = max(1, int(n_distinct_elements / n_buckets)) # take max to avoid div by zero

    # estimate the median 
    est_median = estimate_median(counts_all, n_hash, n_collisions)

    # TODO: use weighted distinct elements to approximate this
    norm_pred = (scores * np.sum(items)) / np.sum(scores)  

    # resulting estimates returned for each item 
    item_est = np.zeros(len(items))
    for i in range(len(items)):
        if np.abs(norm_pred[i] - est_median) <= t*cs_item_std[i]:
            item_est[i] = norm_pred[i] # select the oracle prediction 
        else:
            item_est[i] = cs_item_est[i] # select the CS prediction 
    

    return item_est


def learned_count_sketch_partitions(items, scores, space, partitions):
    ''' 
    1) partitions the items into partitions according to the 'partitions' parameter (boundries) and the oracle scores 
    2) runs count-min on each partition and keep track of number of elements in total 
    3) output count-min estimate for an item if it's within some standard deviations from the mean, o/w output median 
    '''

    # resulting estimates returned for each item 
    item_est = np.zeros(len(items))

    ######################################################
    # 1) partition the items based on the oracle predictions
    ######################################################
    n_partitions = len(partitions)
    loss_per_partition = np.zeros(n_partitions)
    
    # sizes contains the number of elements to place into each partition 
    sizes = np.zeros(n_partitions, dtype=int)
    high = np.max(scores)
    for i in range(n_partitions-1):
        low = partitions[i]
        mask = np.where(np.logical_and(scores>low, scores<=high))
        sizes[i] = len(scores[mask])
        high = low

    # last partition gets everything that's left 
    # there could be a mismatch in test vs. validation preds 
    # so we just dump everything that's lower than the threshold
    # frequency into the last partition 
    sizes[len(sizes)-1] = len(scores[scores <= high])

    # count min setup 
    n_hash_partition_cmin = COUNT_MIN_OPTIMAL_N_HASH
    n_buckets_partition_cmin = int(space / n_hash_partition_cmin / (n_partitions))
    n_buckets_partition_cmin -= int(EXTRA_SPACE_PER_PARTITION_IN_BYTES / (n_hash_partition_cmin * 4)) # space is in units of 4 bytes 

    logger.info("///////////////////////////////////////////////////////////")
    logger.info("Partition sizes:    " + str(sizes))
    logger.info("Partition buckets:  " + str(n_buckets_partition_cmin))
    logger.info("///////////////////////////////////////////////////////////")

    ####################################
    # SANITY CHECKING
    space_total_sanity_check = int(EXTRA_SPACE_PER_PARTITION_IN_BYTES / (n_hash_partition_cmin * 4))
    number_of_items_processed_sanity_check = 0
    sum_indices_sanity_check = 0
    ####################################

    #################################################
    # 2) run count-min on all other partitions 
    #################################################
    start = 0 # partition start index in the items list
    end = 0 # partition end index
    for i in range(n_partitions):
        space_total_sanity_check += n_buckets_partition_cmin*n_hash_partition_cmin

        if sizes[i] == 0:
            continue

        start = end
        end = start + sizes[i]
        part_items = items[start:end]
        part_scores = scores[start:end]

        # uses N_REGISTERS_FOR_HLL bytes (each regiseter is 1 byte) of memory to store distinct elements 
        n_distinct_elements = max(1, hyperloglogsimulate(len(part_items), N_REGISTERS_FOR_HLL)) # take max to avoid division by zero

        bucket_counts, item_to_bucket = random_hash(part_items, n_buckets_partition_cmin)
        cmin_estimates = [bucket_counts[item_to_bucket[j]] for j in range(len(part_items))] # cmin estimates for each value in partition

        # stats about the partition 
        n_collisions = max(1, int(n_distinct_elements / n_buckets_partition_cmin)) # take max to avoid div by zero
        part_mean = np.sum(part_items) / n_distinct_elements
        part_median = estimate_median(bucket_counts, n_hash_partition_cmin, n_collisions) # estimate for the median 
      
        # uses N_BYTES_FOR_SECOND_MOMENT_ESTIMATION bytes of memory to compute second moment 
        part_variance = second_moment_estimate(part_items)/n_distinct_elements - part_mean**2
        part_variance = max(0, part_variance) # avoid negative variance in estimate...
        part_std = math.sqrt(part_variance) 
 
        # compute estimates for this partition
        for j in range(len(part_items)): 
            item_est[start+j] = cmin_estimates[j] / n_collisions

            if np.abs(item_est[start+j] - part_items[j]) > 100:         
                logger.info("partition std        " + str(part_std))
                logger.info("partition mean       " + str(part_mean))
                logger.info("sketch_collisions    " + str(n_collisions))
                logger.info("min (uncorrected)    " + str(cmin_estimates[j]))
                logger.info("min (corrected)      " + str(item_est[start+j]))
                logger.info("actual count:        " + str(part_items[j]))                 

            # compute loss within the partition
            number_of_items_processed_sanity_check += 1
            loss_per_partition[i] += np.abs(item_est[start+j] - part_items[j])
            sum_indices_sanity_check += start+j

        logger.info("///////////////////////////////////////////////////////////")
        logger.info("partition size:               " + str(len(part_items)))
        logger.info("patition size (estimated):    " + str(n_distinct_elements))
        logger.info("partition mean:               " + str(np.mean(part_items)))
        logger.info("partition mean (estimated):   " + str(part_mean))
        logger.info("partition median:             " + str(np.median(part_items)))
        logger.info("partition median (estimated): " + str(part_median))
        logger.info("partition std:                " + str(np.std(part_items)))
        logger.info("partition std (estimated):    " + str(part_std))
        logger.info("n_buckets:                    " + str(n_buckets_partition_cmin))
        logger.info("L1 loss (total):              " + str(loss_per_partition[i]))
        logger.info("///////////////////////////////////////////////////////////")

    logger.info("///////////////////////////////////////////////////////////")
    logger.info("Total space used:  " + str(space_total_sanity_check))
    logger.info("Total loss:        " + str(np.sum(loss_per_partition)))
    logger.info("# of partitions:   " + str(len(sizes)))
    logger.info("# items processed: " + str(number_of_items_processed_sanity_check)  + " (" + str(len(items)) + " total items)")
    logger.info("///////////////////////////////////////////////////////////")


    # make sure we're not using more buckets than originally allocated to the algorithm
    assert space_total_sanity_check <= space, 'ERROR: too much used space; are all the parameters correct?' + ' used: ' + str(space_total_sanity_check) + ' alloted: ' + str(space)
    assert number_of_items_processed_sanity_check == len(items), 'ERROR: did not process all items!'
    assert sum_indices_sanity_check == np.sum(range(len(items))), 'ERROR: did not process all indices!'

    return item_est, loss_per_partition


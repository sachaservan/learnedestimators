import os
import sys
import time
import argparse
import random
import math
import numpy as np
from sketches import count_min, count_sketch
from sketch_common import hyperloglogsimulate, second_moment_estimate

# constants used in experiments 
from experiment_constants import *

# setup logging 
import logging
logger = logging.getLogger('learned_estimators_log')

def learned_count_sketch_partitions(items, scores, space_cs, space_cmin, partitions):
    '''
    learned_count_min_sketch uses the frequency prediction oracle to partition the stream; 
    uses count-sketch for the first partition and count_min sketch (with correction)
    for all other partitions
    '''   

    # resulting estimates returned for each item 
    item_est = np.zeros(len(items))

    # count sketch setup
    n_hash_partition_cs = COUNT_SKETCH_OPTIMAL_N_HASH # found to be optimal 
    n_buckets_partition_cs = int(space_cs / n_hash_partition_cs) 

    ######################################################
    # partition the items based on the oracle predictions
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

    # last partition gets verything that's left 
    # there could be a mismatch in test vs. validation preds 
    # so we just dump everything that's lower than the threshold
    # frequency into the last partition 
    sizes[n_partitions-1] = len(scores[scores <= high])

    # count min setup 
    n_hash_partition_cmin = COUNT_MIN_OPTIMAL_N_HASH
    n_buckets_partition_cmin = int(space_cmin / n_hash_partition_cmin / (n_partitions - 1))
    n_buckets_partition_cmin -= int(EXTRA_SPACE_PER_PARTITION_IN_BYTES / (n_hash_partition_cmin * 4)) # space is in units of 4 bytes 

    logger.info("///////////////////////////////////////////////////////////")
    logger.info("Partition sizes:                       " + str(sizes))
    logger.info("Partition buckets (count-sketch):      " + str(n_buckets_partition_cs))
    logger.info("Partition buckets (count-min sketch):  " + str(n_buckets_partition_cmin))
    logger.info("///////////////////////////////////////////////////////////")

    ####################################
    # SANITY CHECKING
    space_total_sanity_check = 0
    number_of_items_processed_sanity_check = 0
    sum_indices_sanity_check = 0
    ####################################

    start = 0 # partition start index in the items list
    end = sizes[0] # partition end index
    
    #################################################
    # run count sketch on the first partition 
    #################################################
    part_items = items[start:end]
    item_est_cs = count_sketch(part_items, n_buckets_partition_cs, n_hash_partition_cs)
    for i in range(len(part_items)):
        item_est[i] = item_est_cs[i]
        loss_per_partition[0] += np.abs(items[i] - item_est[i])
        number_of_items_processed_sanity_check += 1

    space_total_sanity_check += n_buckets_partition_cs * n_hash_partition_cs

    #################################################
    # run count-min on all other partitions 
    #################################################
    for i in range(1, n_partitions):
        if sizes[i] == 0:
            space_total_sanity_check += n_buckets_partition_cmin*n_hash_partition_cmin
            continue

        start = end
        end = start + sizes[i]
        part_items = items[start:end]
        part_scores = scores[start:end]

        # uses N_REGISTERS_FOR_HLL bytes (each regiseter is 1 byte) of memory to store distinct elements 
        n_distinct_elements = hyperloglogsimulate(len(part_items), N_REGISTERS_FOR_HLL)
        part_mean = np.sum(part_items) / n_distinct_elements

        # uses N_BYTES_FOR_SECOND_MOMENT_ESTIMATION bytes of memory to compute second moment 
        part_std = math.sqrt(second_moment_estimate(part_items)/n_distinct_elements - part_mean**2) 
        cmin_estimates = count_min(part_items, n_buckets_partition_cmin, n_hash_partition_cmin)

        expected_collisions = int(n_distinct_elements / n_buckets_partition_cmin)

        # compute estimates for this partition
        for j in range(len(part_items)): 
            # treshold correction based on # collisions         
            if (cmin_estimates[j] - expected_collisions*part_mean) >= part_std:
                item_est[start+j] = max(0, cmin_estimates[j]  - (expected_collisions - 1)*part_mean)
            else:
                item_est[start+j] = int(round(cmin_estimates[j] / expected_collisions))

            # if np.abs(item_est[start+j] - part_items[j]) > 100:         
            #     logger.info()
            #     logger.info("partition size       " + str(len(splits[i])))
            #     logger.info("partition std        " + str(part_std))
            #     logger.info("partition mean       " + str(part_mean))
            #     logger.info("sketch_collisions    " + str(sketch_collisions))
            #     logger.info("estimates            " + str(sketch_estimates))
            #     logger.info("min (uncorrected)    " + str(np.min(sketch_estimates)))x
            #     logger.info("min (corrected)      " + str(item_est[start+j]))
            #     logger.info("actual count:        " + str(part_items[j]))                 

            # compute loss within the partition
            number_of_items_processed_sanity_check += 1
            loss_per_partition[i] += np.abs(item_est[start+j] - part_items[j])
            sum_indices_sanity_check += start+j

        logger.info("///////////////////////////////////////////////////////////")
        logger.info("partition size:               " + str(len(part_items)))
        logger.info("patition size (estimated):    " + str(n_distinct_elements))
        logger.info("partition mean:               " + str(np.mean(part_items)))
        logger.info("partition mean (estimated):   " + str(part_mean))
        logger.info("partition std:                " + str(np.std(part_items)))
        logger.info("partition std (estimated):    " + str(part_std))
        logger.info("n_buckets:                    " + str(n_buckets_partition_cmin))
        logger.info("L1 loss (total):              " + str(loss_per_partition[i]))
        logger.info("///////////////////////////////////////////////////////////")

    # make sure we're not using more buckets than originally allocated to the algorithm
    if space_total_sanity_check > space_cmin + space_cs:
        logger.info("WARNING: too much used space; are all the parameters correct?")
    
    logger.info("///////////////////////////////////////////////////////////")
    logger.info("Total space used:  " + str(space_total_sanity_check))
    logger.info("Total loss:        " + str(np.sum(loss_per_partition)))
    logger.info("# of partitions:   " + str(len(sizes)))
    logger.info("# items processed: " + str(number_of_items_processed_sanity_check)  + " (" + str(len(items)) + " total items)")
    logger.info("///////////////////////////////////////////////////////////")

    return item_est, loss_per_partition


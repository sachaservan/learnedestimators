import random
import numpy as np
from sketch_common import random_hash, random_hash_with_sign

# setup logging 
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def count_min(items, n_buckets, n_hash):
    ''' 
    regular count min algorithm 
    '''
    counts_all = np.zeros((n_hash, n_buckets))
    item_buckets_all = np.zeros((n_hash, len(items)), dtype=int)
    for i in range(n_hash):
        counts, item_buckets = random_hash(items, n_buckets)
        counts_all[i] = counts
        item_buckets_all[i] = item_buckets

    item_est = np.zeros(len(items))
    for i in range(len(items)): 
        sketch_estimates = [counts_all[k, item_buckets_all[k, i]] for k in range(n_hash)]
        item_est[i] = np.min(sketch_estimates)

    return item_est

def count_sketch(y, n_buckets, n_hash):
    ''' 
    regular count sketch algorithm 
    '''
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
        item_est[i] = np.median([y_signs_all[k, i] * counts_all[k, y_buckets_all[k, i]] for k in range(n_hash)])
    
    return item_est


def median_sketch(y, n_buckets, n_hash):
    ''' 
    regular count sketch algorithm 
    '''
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
        item_est[i] = np.median([y_signs_all[k, i] * counts_all[k, y_buckets_all[k, i]] for k in range(n_hash)])

    
    return np.median(item_est)

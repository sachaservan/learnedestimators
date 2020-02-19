import os
import sys
import time
import csv
import argparse
import numpy as np
from itertools import repeat
from multiprocessing import Pool
from utils import get_data, get_stat, git_log, feat_to_string, get_data_str_with_ports_list
from sketches import count_min, count_sketch
from sketch_utils import order_y_wkey_list
from learned_sketches import learned_count_sketch, learned_count_min_sketch
from aol_utils import get_data_aol_query_list

def loss_weighted(y_true, y_est):
    return np.abs(y_true - y_est) * y_true

def loss_l1(y_true, y_est):
    return np.abs(y_true - y_est)

def loss_l2(y_true, y_est):
    return np.abs(y_true - y_est) ** 2

# the loss function used for the evaluation
# TODO(sss): make this a parameter
loss_function = loss_l2

def run_count_min_sketch(y, n_hashes, n_buckets, name):
    loss = count_min(y, n_buckets, n_hashes, loss_function)
    print('%s: # hashes %d, # buckets %d - loss %.2f\t time: %.2f sec' % \
        (name, n_hashes, n_buckets, loss, time.time() - start_t))
    return loss

def run_count_sketch(y, n_hashes, n_buckets, name):
    loss = count_sketch(y, n_buckets, n_hashes, loss_function)
    print('%s: # hashes %d, # buckets %d - loss %.2f\t time: %.2f sec' % \
        (name, n_hashes, n_buckets, loss, time.time() - start_t))
    return loss

def run_learned_count_min_sketch(y, y_scores, n_hashes, n_buckets, name):
    start_t = time.time()
    loss = learned_count_min_sketch(y, y_scores, n_hashes, n_buckets, loss_function)
    print('%s: # hashes %d, # buckets %d - loss %.2f\t time: %.2f sec' % \
        (name, n_hashes, n_buckets, loss, time.time() - start_t))
    return loss

def run_learned_sketch(y, y_scores, n_hashes, n_buckets, name):
    start_t = time.time()
    loss = learned_count_sketch(y, y_scores, n_hashes, n_buckets, loss_function)
    print('%s: # hashes %d, # buckets %d - loss %.2f\t time: %.2f sec' % \
        (name, n_hashes, n_buckets, loss, time.time() - start_t))
    return loss

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--test_results", type=str, nargs='*', help="testing results of a model (.npz file)", default='')
    argparser.add_argument("--valid_results", type=str, nargs='*', help="validation results of a model (.npz file)", default='')
    argparser.add_argument("--test_data", type=str, nargs='*', help="list of input .npy data", required=True)
    argparser.add_argument("--valid_data", type=str, nargs='*', help="list of input .npy data", required=True)
    argparser.add_argument("--lookup_data", type=str, nargs='*', help="list of input .npy data", default=[])
    argparser.add_argument("--save", type=str, help="prefix to save the results", required=True)
    argparser.add_argument("--seed", type=int, help="random state for sklearn", default=69)
    argparser.add_argument("--space_list", type=float, nargs='*', help="space in MB", default=[])
    argparser.add_argument("--n_hashes_list", type=int, nargs='*', help="number of hashes", required=True)
    argparser.add_argument("--n_workers", type=int, help="number of workers", default=10)
    argparser.add_argument("--aol_data", action='store_true', default=False)
    args = argparser.parse_args()

    name = 'learned_count_min'
 
    command = ' '.join(sys.argv) + '\n'
    log_str = command
    log_str += git_log() + '\n'
    print(log_str)
    np.random.seed(args.seed)

    folder = os.path.join('param_results', name, '')
    if not os.path.exists(folder):
        os.makedirs(folder)

    start_t = time.time()
    if args.aol_data:
        x_valid, y_valid = get_data_aol_query_list(args.valid_data)
        x_test, y_test = get_data_aol_query_list(args.test_data)
    else:
        x_valid, y_valid = get_data_str_with_ports_list(args.valid_data)
        x_test, y_test = get_data_str_with_ports_list(args.test_data)
    log_str += get_stat('valid data:\n'+'\n'.join(args.valid_data), x_valid, y_valid)
    log_str += get_stat('test data:\n'+'\n'.join(args.test_data), x_test, y_test)

    log_str += ('data loading time: %.1f sec' % (time.time() - start_t)) + "\n"
    print(log_str)

    if args.valid_results:
        key = 'valid_output'
        y_valid_ordered, y_valid_scores = order_y_wkey_list(y_valid, args.valid_results, key)

    if args.test_results:
        key = 'test_output'
        y_test_ordered, y_test_scores = order_y_wkey_list(y_test, args.test_results, key)

    # testing parameters 
    n_hashes = []
    n_buckets = []
    for space in args.space_list:
        for n_hash in args.n_hashes_list:
            space_to_buckets = int(space * 1e6 / (n_hash * 4))
            n_hashes.append(n_hash)
            n_buckets.append(space_to_buckets)

    #################################################################
    # evaluate vanilla sketching algorithm against learned sketches
    #################################################################

    # learned count min sketch
    pool = Pool(args.n_workers)
    test_results_learned = pool.starmap(
        run_learned_count_min_sketch, zip(repeat(y_valid_ordered), repeat(y_valid_scores), 
        n_hashes, n_buckets, repeat('learned_count_min_sketch')))
    pool.close()
    pool.join()

    # vanilla count min sketch
    pool = Pool(args.n_workers)
    test_results = pool.starmap(
        run_count_min_sketch, zip(repeat(y_valid_ordered), 
        n_hashes, n_buckets, repeat('count_min_sketch')))
    pool.close()

    log_str += "Percentage improvement (loss_regular / loss_learned): \n"
    for i in range(len(test_results)):
        percentage = test_results[i] / test_results_learned[i]
        log_str += (" " + str(np.floor((percentage - 1.0) * 100))) + "\n"

    print(log_str)

    with open(os.path.join(folder, args.save+'.log'), 'w') as f:
        f.write(log_str)

    np.savez(os.path.join(folder, args.save+'learned_count_min_results'),
        command=command,
        loss_vanilla=test_results,
        loss_learned=test_results_learned,
        n_hashes=n_hashes,
        n_buckets=n_buckets,
        space_list=args.space_list,
    )
    
    # learned count sketch
    pool = Pool(args.n_workers)
    test_results_learned = pool.starmap(
        run_learned_sketch, zip(repeat(y_valid_ordered), repeat(y_valid_scores), 
        n_hashes, n_buckets, repeat('learned_count_sketch')))
    pool.close()

    # vanilla count sketch
    pool = Pool(args.n_workers)
    test_results = pool.starmap(
        run_count_sketch, zip(repeat(y_valid_ordered), 
        n_hashes, n_buckets, repeat('count_sketch')))
    pool.close()

    log_str += "Percentage improvement (loss_regular / loss_learned): \n"
    for i in range(len(test_results)):
        percentage = test_results[i] / test_results_learned[i]
        log_str += (" " + str(np.floor((percentage - 1.0) * 100))) + "\n"

    print(log_str)

    np.savez(os.path.join(folder, args.save+'learned_sketch_results'),
        command=command,
        loss_vanilla=test_results,
        loss_learned=test_results_learned,
        n_hashes=n_hashes,
        n_buckets=n_buckets,
        space_list=args.space_list,
    )
  
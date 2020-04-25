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
from learned_sketches import  learned_count_sketch, learned_count_sketch_just_cutoff
from aol_utils import get_data_aol_query_list

def loss_weighted(loss, y_true, y_est):
    return loss + np.abs(y_true - y_est) * y_true

def loss_l1(loss, y_true, y_est):
    return loss + np.abs(y_true - y_est)

def loss_l2(loss, y_true, y_est):
    return loss + np.abs(y_true - y_est) ** 2

def loss_hinge(loss, y_true, y_est):
    return loss + max(0, y_true - y_est + 1)

# the loss function used for the evaluation
# TODO(sss): make this a parameter
loss_function = loss_l2


def run_count_sketch(y, n_hashes, n_buckets, name):
    start_t = time.time()
    _, loss, _ = count_sketch(y, n_buckets, n_hashes, loss_function=loss_function)
    print('%s: # hashes %d, # buckets %d - loss %.0f\t time: %.2f sec' % \
        (name, n_hashes, n_buckets, loss, time.time() - start_t))
    return loss / len(y)

def run_learned_count_sketch(y, y_scores, n_hashes, n_buckets, name, cutoff):
    start_t = time.time()
    loss, percent_oracle = learned_count_sketch(y, y_scores, n_hashes, n_buckets, loss_function, cutoff)
    print('%s: # hashes %d, # buckets %d - loss %.0f\t time: %.2f sec' % \
        (name, n_hashes, n_buckets, loss, time.time() - start_t))
    return loss / len(y), percent_oracle


def run_learned_count_sketch_just_cutoff(y, y_scores, n_hashes, n_buckets, name):
    start_t = time.time()
    loss = learned_count_sketch_just_cutoff(y, y_scores, n_hashes, n_buckets, loss_function)
    print('%s: # hashes %d, # buckets %d - loss %.0f\t time: %.2f sec' % \
        (name, n_hashes, n_buckets, loss, time.time() - start_t))
    return loss / len(y)



def load_dataset(dataset, model, is_aol=False):
    start_t = time.time()
    print("Loading dataset...")
    if is_aol:
        x_valid, y_valid = get_data_aol_query_list(dataset)
    else:
        x_valid, y_valid = get_data_str_with_ports_list(dataset)
    print("done.")
    
    print('Data loading time: %.1f sec' % (time.time() - start_t))

    print("Loading model...")
    key = 'valid_output'
    data_sorted, oracle_scores_sorted = order_y_wkey_list(y_valid, model, key)
    print("done.")

    # print("data:   " + str(data_sorted))
    # print("scores: " + str(oracle_scores_sorted))

    return data_sorted, oracle_scores_sorted


def experiment_comapre_loss(
    data,
    oracle_scores,
    n_hashes, 
    n_buckets,
    n_workers, 
    space_list, 
    save_folder, 
    save_file, 
    run_regular_count_sketch=False,
    run_learned_experiment=False, 
    run_cutoff_experiment=False, 
    run_perfect_oracle_experiment=False):

    #################################################################
    # evaluate vanilla sketching algorithm against learned sketches
    #################################################################
    test_results_learned_no_cutoff = []
    percent_oracle_no_cutoff = []
    if run_learned_experiment:
        print("Running learned count sketch")
        # learned algorithm with no cutoff 
        pool = Pool(n_workers)
        results = pool.starmap(
            run_learned_count_sketch, zip(repeat(data), repeat(oracle_scores), 
            n_hashes, n_buckets, repeat('learned_count_sketch'), repeat(False)))
        pool.close()
        pool.join()

        test_results_learned_no_cutoff = np.array([x[0] for x in results])
        percent_oracle_no_cutoff = np.array([x[1] for x in results])

    # learned algo with cutoff 
    test_results_learned_cutoff = []
    percent_oracle_cutoff = []
    if run_cutoff_experiment:
        print("Running learned count sketch with cutoff")
        # learned algorithm with cutoff 
        pool = Pool(n_workers)
        results = pool.starmap(
            run_learned_count_sketch, zip(repeat(data), repeat(oracle_scores), 
            n_hashes, n_buckets, repeat('learned_count_sketch+cutoff'), repeat(True)))
        pool.close()
        pool.join()

        test_results_learned_cutoff = np.array([x[0] for x in results])
        percent_oracle_cutoff = np.array([x[1] for x in results])

    # vanilla sketch + cutoff 
    test_results_just_cutoff = []
    if run_cutoff_experiment:
        print("Running vanilla count sketch with cutoff")
        # count sketch algorithm with cutoff 
        pool = Pool(n_workers)
        test_results_just_cutoff = pool.starmap(
            run_learned_count_sketch_just_cutoff, zip(repeat(data), repeat(oracle_scores), 
            n_hashes, n_buckets, repeat('count_sketch+cutoff')))
        pool.close()
        pool.join()


    # learned algo with a perfect prediction oracle
    test_results_learned_perfect_oracle = []
    percent_oracle_perfect = []
    if run_perfect_oracle_experiment:
        print("Running learned count sketch with perfect oracle")
        # learned algorithm with cutoff 
        pool = Pool(n_workers)

        # set scores = data (i.e. true counts)
        results = pool.starmap(
            run_learned_count_sketch, zip(repeat(data), repeat(data), 
            n_hashes, n_buckets, repeat('learned_count_sketch+perfect_oracle'), repeat(False)))
        pool.close()
        pool.join()

        test_results_learned_perfect_oracle = np.array([x[0] for x in results])
        percent_oracle_perfect = np.array([x[1] for x in results])
   
    # vanilla count sketch
    test_results = []
    if run_regular_count_sketch:
        print("Running vanilla count sketch")
        pool = Pool(n_workers)
        test_results = pool.starmap(
            run_count_sketch, zip(repeat(data), 
            n_hashes, n_buckets, repeat('count_sketch')))
        pool.close()


    #################################################################
    # print results comparing the learned algorithm to the regular count sketch
    #################################################################
    if run_regular_count_sketch and run_learned_experiment:
        print("Percentage improvement (loss_regular / loss_learned):")
        for i in range(len(test_results)):
            percentage = test_results[i] / test_results_learned_no_cutoff[i]
            print(" " + str(percentage))

    if run_regular_count_sketch and run_cutoff_experiment:
        print("Percentage improvement w/ cutoff (loss_regular / loss_learned):")
        for i in range(len(test_results)):
            percentage = test_results[i] / test_results_learned_cutoff[i]
            print(" " + str(percentage))


    #################################################################
    # save all results to the folder
    #################################################################
    np.savez(os.path.join(save_folder, save_file),
        loss_vanilla=test_results,
        loss_learned_no_cutoff=test_results_learned_no_cutoff,
        loss_learned_cutoff=test_results_learned_cutoff,
        loss_learned_perfect_oracle=test_results_learned_perfect_oracle,
        loss_just_cutoff=test_results_just_cutoff,
        percent_oracle_no_cutoff=percent_oracle_no_cutoff,
        percent_oracle_cutoff=percent_oracle_cutoff,
        percent_oracle_perfect=percent_oracle_perfect,
        n_hashes=n_hashes,
        n_buckets=n_buckets,
        space_list=space_list)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--model", type=str, nargs='*', default='', help="validation results of a model (.npz file)")
    argparser.add_argument("--dataset", type=str, nargs='*', required=True,  help="list of input .npy data")
    argparser.add_argument("--save_folder", type=str, required=True, help="folder to save the results in")
    argparser.add_argument("--save_file", type=str, required=True, help="prefix to save the results")
    argparser.add_argument("--seed", type=int, default=420, help="random state for sklearn")
    argparser.add_argument("--space_list", type=float, nargs='*', default=[], help="space in MB")
    argparser.add_argument("--n_hashes_list", type=int, nargs='*', required=True, help="number of hashes")
    argparser.add_argument("--n_workers", type=int, default=10, help="number of worker threads",)
    argparser.add_argument("--aol_data", action='store_true', default=False)
    argparser.add_argument("--run_cutoff_version", action='store_true', default=False)
    argparser.add_argument("--run_learned_version", action='store_true', default=False)
    argparser.add_argument("--run_perfect_oracle_version", action='store_true', default=False)
    argparser.add_argument("--run_regular_count_sketch", action='store_true', default=False)
    args = argparser.parse_args()

    name = 'learned_count_sketch'

    # set the random seed for numpy values
    np.random.seed(args.seed)

    # output folder for storing the results of the experiment
    folder = os.path.join(args.save_folder, name, '')
    if not os.path.exists(folder):
        os.makedirs(folder)

    # load the dataset
    data, oracle_scores = load_dataset(
        args.dataset, 
        args.model, 
        args.aol_data
    )

    # testing parameters 
    n_hashes = []
    n_buckets = []
    for space in args.space_list:
        for n_hash in args.n_hashes_list:
            space_to_buckets = int(space * 1e6 / (n_hash * 4))
            n_hashes.append(n_hash)
            n_buckets.append(space_to_buckets)

    print("Experiment parameters: ")
    print("#hashes:" + str(n_hash))
    print("#buckets: " + str(n_buckets))

    # run the experiment with the specified parameters
    experiment_comapre_loss(
        data, 
        oracle_scores,
        n_hashes, 
        n_buckets,
        args.n_workers, 
        args.space_list, 
        args.save_folder, 
        args.save_file, 
        run_regular_count_sketch=args.run_regular_count_sketch,
        run_learned_experiment=args.run_learned_version,
        run_cutoff_experiment=args.run_cutoff_version, 
        run_perfect_oracle_experiment=args.run_perfect_oracle_version)
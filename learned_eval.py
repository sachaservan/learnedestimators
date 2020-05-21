import os
import sys
import time
import csv
import argparse
import numpy as np
from itertools import repeat
from multiprocessing import Pool
from utils import get_data, get_stat, git_log, feat_to_string, get_data_str_with_ports_list
from sketches import count_min, count_sketch, log_count_min
from sketch_utils import order_y_wkey_list
from weighted_distinct_elements import compute_sum_estimate
from learned_sketches import  learned_count_sketch_just_cutoff, learned_count_sketch_partitions
from aol_utils import get_data_aol_query_list

def run_count_sketch(y, n_hashes, n_buckets, name):
    estimates = count_sketch(y, n_buckets, n_hashes)
    return estimates

def run_learned_count_sketch(y, y_scores, n_hashes, n_buckets, name, cutoff): 
    estimates, loss_per_partition = learned_count_sketch_partitions(y.copy(), y_scores.copy(), n_hashes, n_buckets, cutoff)
    return estimates, loss_per_partition

def run_learned_count_sketch_just_cutoff(y, y_scores, n_hashes, n_buckets, name):
    loss = learned_count_sketch_just_cutoff(y, y_scores, n_hashes, n_buckets)
    return loss


def load_dataset(dataset, model, is_aol=False, is_synth=False):

    if is_synth:
        N = 200000
        a = 1.2 # zipf param
        data = np.random.zipf(2, N).flatten()
        error = np.random.normal(0, 20, N).flatten()
        scores =  data + error
        scores += np.abs(np.min(scores)) + 1
        sort = np.argsort(scores)[::-1]
        data = data[sort]
        scores = scores[sort]

        return data, scores

    start_t = time.time()
    print("Loading dataset...")
    if is_aol:
        x_valid, y_valid = get_data_aol_query_list(dataset)
    else:
        x_valid, y_valid = get_data_str_with_ports_list(dataset)
    print("done. Number of items: " + str(len(y_valid)))
    
    print('Data loading time: %.1f sec' % (time.time() - start_t))

    print("Loading model...")
    key = 'valid_output'
    data, oracle_scores = order_y_wkey_list(y_valid, model, key)
    if not is_aol:
        oracle_scores= np.exp(oracle_scores)
    print("done.")


    # make sure all scores are positive 
    minscore = np.min(np.array(oracle_scores))
    if minscore < 0:
        minscore = np.abs(minscore) + 1
        oracle_scores = np.array([x + minscore for x in oracle_scores])


    sort = np.argsort(oracle_scores)[::-1]
    oracle_scores_sorted = oracle_scores[sort]
    data_sorted = data[sort]

    print("# items: " + str(len(data_sorted)))
    print("data:    " + str(data_sorted))
    print("scores:  " + str(oracle_scores_sorted))

    return data_sorted, oracle_scores_sorted

def experiment_oracle_accuracy(data, scores):
    
    print("--------------------------------")
    print("actual counts:    " + str(data))
    print("predicted counts: " + str(scores))
    print("--------------------------------")

    return data, scores

def experiment_comapre_loss(
    data,
    oracle_scores,
    n_hashes, 
    n_buckets,
    n_workers, 
    space_list, 
    save_folder, 
    save_file, 
    use_count_sketch_results_file=False,
    run_regular_count_sketch=False,
    run_learned_experiment=False, 
    run_cutoff_experiment=False, 
    run_perfect_oracle_experiment=False):

    #################################################################
    # compare oracle predictions to actual data
    #################################################################
    print("Evaluating oracle accuracy...")
    true_values, oracle_predictions = experiment_oracle_accuracy(data, oracle_scores)
    print("Done.")

    # sort the data however we need it 
    sort = np.argsort(oracle_predictions)[::-1]
    oracle_predictions = oracle_predictions[sort]
    data = data[sort]
    true_values = true_values[sort]

    #################################################################
    # evaluate vanilla sketching algorithm against learned sketches
    #################################################################
    test_algo_predictions = []
    loss_per_partition = []
    if run_learned_experiment:
        print("Running learned count sketch")
        # learned algorithm with no cutoff 
        pool = Pool(n_workers)
        results = pool.starmap(
            run_learned_count_sketch, zip(repeat(data), repeat(oracle_predictions), 
            n_hashes, n_buckets, repeat('count_sketch'), repeat(False)))
        pool.close()
        pool.join()
        test_algo_predictions = [x[0] for x in results]
        loss_per_partition = [x[1] for x in results]
        print("L1 loss " + str(np.sum(np.abs(data - test_algo_predictions))))
        print("L2 loss " + str(np.sum(np.abs(data - test_algo_predictions)**2)))


    # learned algo with cutoff 
    test_results_learned_cutoff = []
    if run_cutoff_experiment:
        print("Running learned count sketch with cutoff")
        # learned algorithm with cutoff 
        pool = Pool(n_workers)
        test_results_learned_cutoff = pool.starmap(
            run_learned_count_sketch, zip(repeat(data), repeat(oracle_predictions), 
            n_hashes, n_buckets, repeat('learned_count_sketch+cutoff'), repeat(True)))
        pool.close()
        pool.join()
        

    # vanilla sketch + cutoff 
    test_results_just_cutoff = []
    if run_cutoff_experiment:
        print("Running vanilla count sketch with cutoff")
        # count sketch algorithm with cutoff 
        pool = Pool(n_workers)
        test_results_just_cutoff = pool.starmap(
            run_learned_count_sketch_just_cutoff, zip(repeat(data), repeat(oracle_predictions), 
            n_hashes, n_buckets, repeat('count_sketch+cutoff')))
        pool.close()
        pool.join()

    # vanilla count sketch
    test_count_sketch_predictions = []
    if run_regular_count_sketch:

        if use_count_sketch_results_file:
            count_sketch_res = np.load(save_folder + "/" + save_file + "count_sketch_results.npz")
            test_count_sketch_predictions = count_sketch_res['test_results_count_sketch']
        else:
            print("Running vanilla count sketch")
            pool = Pool(n_workers)
            test_count_sketch_predictions = pool.starmap(
                run_count_sketch, zip(repeat(data), n_hashes, n_buckets, repeat('count_sketch')))
            pool.close()
            pool.join()
        
            # save the results for future use
            np.savez(os.path.join(save_folder, save_file + "count_sketch_results"),
            test_results_count_sketch=test_count_sketch_predictions)

    #################################################################
    # save all results to the folder
    #################################################################
    np.savez(os.path.join(save_folder, save_file),
        true_values=true_values,
        oracle_predictins=oracle_predictions,
        test_algo_predictions=test_algo_predictions,
        test_algo_predictions_cutoff= test_results_learned_cutoff,
        test_count_sketch_predictions=test_count_sketch_predictions,
        test_count_sketch_predictions_cutoff=test_results_just_cutoff,
        test_loss_per_partition=loss_per_partition,
        n_hashes=n_hashes,
        n_buckets=n_buckets,
        space_list=space_list)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--model", type=str, nargs='*', default='', help="validation results of a model (.npz file)")
    argparser.add_argument("--dataset", type=str, nargs='*',  help="list of input .npy data")
    argparser.add_argument("--count_sketch_results", type=str,  help="results for count sketch .npz file")
    argparser.add_argument("--save_folder", type=str, required=True, help="folder to save the results in")
    argparser.add_argument("--save_file", type=str, required=True, help="prefix to save the results")
    argparser.add_argument("--seed", type=int, default=42, help="random state for sklearn")
    argparser.add_argument("--space_list", type=float, nargs='*', default=[], help="space in MB")
    argparser.add_argument("--n_hashes_list", type=int, nargs='*', required=True, help="number of hashes")
    argparser.add_argument("--n_workers", type=int, default=10, help="number of worker threads",)
    argparser.add_argument("--use_count_sketch_results_file", action='store_true', default=False)
    argparser.add_argument("--aol_data", action='store_true', default=False)
    argparser.add_argument("--synth_data", action='store_true', default=False)
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
        args.aol_data,
        args.synth_data)

    # testing parameters 
    n_hashes = []
    n_buckets = []
    for space in args.space_list:
        for n_hash in args.n_hashes_list:
            space_to_buckets = int(space * 1e6 / (n_hash * 4))
            n_hashes.append(n_hash)
            n_buckets.append(space_to_buckets)

    print("--------------------------------")
    print("Experiment parameters: ")
    print(" - #hashes:  " + str(n_hash))
    print(" - #buckets: " + str(n_buckets))
    print("--------------------------------")

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
        use_count_sketch_results_file=args.use_count_sketch_results_file,
        run_regular_count_sketch=args.run_regular_count_sketch,
        run_learned_experiment=args.run_learned_version,
        run_cutoff_experiment=args.run_cutoff_version, 
        run_perfect_oracle_experiment=args.run_perfect_oracle_version)
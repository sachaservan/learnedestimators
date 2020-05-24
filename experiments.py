import os
import sys
import time
import csv
import argparse
import numpy as np

from halo import Halo
from itertools import repeat
from multiprocessing import Pool, set_start_method, get_context
from sketches import count_min, count_sketch
from learned_sketches import learned_count_sketch_partitions

# utils for parsing the dataset and results 
from utils import get_data_aol_query_list
from utils import get_data, get_stat, git_log, feat_to_string, get_data_str_with_ports_list

# constants used in experiments 
from experiment_constants import *

# setup logging 
import logging
logger = logging.getLogger('learned_estimators_log')

#################################################
# setup logging 
#################################################
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger.propagate = False
logger.addHandler(logging.FileHandler('eval.log', 'w'))

#################################################
# each algorithm variant used in the experiemnts 
#################################################
def run_count_sketch(y, n_hashes, n_buckets):
    estimates = count_sketch(y, n_buckets, n_hashes)
    return estimates

def run_cutoff_count_sketch(y, y_scores, space, cutoff_threshold): 
    y_cutoff = y[cutoff_threshold:] # all items that have a predicted score > cutoff_thresh
    table_estimates = y[:cutoff_threshold] # store exact counts for all 
    n_buckets = int(space / COUNT_MIN_OPTIMAL_N_HASH)
    n_hash = int(space / COUNT_SKETCH_OPTIMAL_N_HASH)
    sketch_estimates = count_sketch(y_cutoff, n_buckets, n_hash)
    
    return np.concatenate((table_estimates, sketch_estimates))
  

def run_learned_count_sketch(y, y_scores, space_cs, space_cmin, partitions, cutoff=False, cutoff_threshold=0): 

    if cutoff:
        y_cutoff = y[cutoff_threshold:] # all items that have a predicted score > cutoff_thresh
        y_scores_cutoff = y_scores[cutoff_threshold:]
        table_estimates = y[:cutoff_threshold] # store exact counts for all 

        sketch_estimates, loss_per_partition = learned_count_sketch_partitions(y_cutoff.copy(), y_scores_cutoff.copy(), space_cs, space_cmin, partitions)
        return np.concatenate((table_estimates, sketch_estimates)), loss_per_partition
    else:
        estimates, loss_per_partition = learned_count_sketch_partitions(y.copy(), y_scores.copy(), space_cs, space_cmin, partitions)
  
    return estimates, loss_per_partition

#################################################
def compute_partitions(scores, space, n_partitions):
    splits = np.array_split(scores, n_partitions)
    sizes = [len(splits[k]) for k in range(n_partitions)]

    n_partitions_cs = 1
    n_items = sizes[0]
    for i in range(len(sizes)):
        if n_items < space:
            n_partitions_cs += 1
            n_items += sizes[i]
        else:
            break

    partitions = np.zeros(n_partitions - n_partitions_cs)
    for i in range(n_partitions_cs, n_partitions):   
        partitions[i - n_partitions_cs] = np.min(splits[i]) # min score as threshold 
    
    return partitions

def find_best_parameters_for_cutoff(test_data, test_oracle_scores, space_list, space_allocations, n_workers, save_folder, save_file):
    cutoff_frac_to_test = CUTOFF_FRAC_TO_TEST
    best_space_cs = []
    best_cutoff_thresh_for_space = []
    for i, test_space in enumerate(space_allocations):
        test_space_cs = []
        test_params_cutoff_thresh = []

        # test all combinations 
        for test_cutoff_frac in cutoff_frac_to_test:
            # combination of parameters to test
            cutoff_thresh = int((test_cutoff_frac * test_space) / CUTOFF_SPACE_COST_FACTOR)
            test_params_cutoff_thresh.append(cutoff_thresh)
            test_space_post_cutoff = int(test_space - cutoff_thresh * CUTOFF_SPACE_COST_FACTOR)
            test_space_cs.append(int(test_space_post_cutoff))

        logger.info("Learning best parameters for space setting...")
        with get_context("spawn").Pool() as pool:
            test_cutoff_predictions = pool.starmap(
                run_cutoff_count_sketch, 
                zip(repeat(test_data), 
                repeat(test_oracle_scores), 
                test_space_cs, 
                test_params_cutoff_thresh))
            pool.close()
            pool.join()

        losses = [np.sum(np.abs(test_data - predictions)**2) for predictions in test_cutoff_predictions]
        best_loss_idx = np.argmin(losses)

        space_cs = test_space_cs[best_loss_idx]
        cutoff_thresh = test_params_cutoff_thresh[best_loss_idx]
        best_space_cs.append(space_cs)
        best_cutoff_thresh_for_space.append(cutoff_thresh)
        
    np.savez(os.path.join(save_folder, save_file),
        space_list=space_list,
        best_space_cs=best_space_cs,
        best_cutoff_thresh_for_space=best_cutoff_thresh_for_space)

    return best_space_cs, best_cutoff_thresh_for_space

def find_best_parameters_for_learned_algo(test_data, test_oracle_scores, space_list, space_allocations, n_workers, save_folder, save_file, cutoff=False):
    space_fractions_to_test = SPACE_FRAC_TO_TEST
    n_partitions_to_test = NUM_PARTITIONS_TO_TEST
    cutoff_frac_to_test = [0] # no cutoff 
    if cutoff:
        cutoff_frac_to_test = CUTOFF_FRAC_TO_TEST 

    # compute partitions for each space allocation 
    best_space_cs = []
    best_space_cmin = []
    best_partitions_for_space = []
    best_cutoff_thresh_for_space = []
    for i, test_space in enumerate(space_allocations):
        test_space_cs = []
        test_space_cmin = []
        test_params_cutoff_thresh = []
        test_params_partitions = []

        # test all combinations 
        for space_frac in space_fractions_to_test:
            for test_n_partition in n_partitions_to_test:
                for test_cutoff_frac in cutoff_frac_to_test:
                    # combination of parameters to test
                    cutoff_thresh = int((test_cutoff_frac * test_space) / CUTOFF_SPACE_COST_FACTOR)
                    test_params_cutoff_thresh.append(cutoff_thresh)
                    test_space_post_cutoff = int(test_space - cutoff_thresh * CUTOFF_SPACE_COST_FACTOR)

                    test_space_cs.append(int(test_space_post_cutoff * space_frac))
                    test_space_cmin.append(int(test_space_post_cutoff * (1.0 - space_frac)))
                    space_for_cs = int(test_space_post_cutoff * space_frac / COUNT_SKETCH_OPTIMAL_N_HASH) # want approx. as many buckets as items so divide by n_hash
                    partitions = compute_partitions(test_oracle_scores, space_for_cs , test_n_partition)
                    test_params_partitions.append(partitions)
                    test_params_cutoff_thresh.append(cutoff_thresh)

        logger.info("Learning best parameters for space setting...")
        with get_context("spawn").Pool() as pool:
            results = pool.starmap(
                run_learned_count_sketch, 
                zip(repeat(test_data), 
                repeat(test_oracle_scores), 
                test_space_cs, 
                test_space_cmin, 
                test_params_partitions, 
                repeat(cutoff), 
                test_params_cutoff_thresh))
            pool.close()
            pool.join()

        test_algo_predictions = [x[0] for x in results]
        losses = [np.sum(np.abs(test_data - predictions)**2) for predictions in test_algo_predictions]
        best_loss_idx = np.argmin(losses)
      
        space_cs = test_space_cs[best_loss_idx]
        space_cmin = test_space_cmin[best_loss_idx]
        partitions = test_params_partitions[best_loss_idx]
        cutoff_thresh = test_params_cutoff_thresh[best_loss_idx]
        best_space_cs.append(space_cs)
        best_space_cmin.append(space_cmin) # 1-space_frac (use all remaining space)
        best_partitions_for_space.append(partitions)
        best_cutoff_thresh_for_space.append(cutoff_thresh)

    logger.info("All L2 losses       " + str(losses))
    logger.info("Best L2 loss        " + str(losses[best_loss_idx]))
    logger.info("Best space cs:      " + str(best_space_cs))
    logger.info("Best space cmin:    " + str(best_space_cmin))
    logger.info("Best partitions:    " + str(partitions))
    logger.info("Best cutoff thresh: " + str(best_cutoff_thresh_for_space))

    np.savez(os.path.join(save_folder, save_file),
        space_list=space_list,
        best_space_cs=best_space_cs,
        best_space_cmin=best_space_cmin,
        best_partitions_for_space=best_partitions_for_space,
        best_cutoff_thresh_for_space=best_cutoff_thresh_for_space,
        space_allocations=space_allocations)

    return best_space_cs, best_space_cmin, best_partitions_for_space

def experiment_comapre_loss_with_cutoff(
    test_data,
    test_oracle_scores,
    valid_data,
    valid_oracle_scores,
    space_alloc,
    n_workers, 
    space_list, 
    save_folder, 
    save_file, 
    run_regular_count_sketch=False,
    run_learned_experiment=False, 
    run_perfect_oracle_experiment=False):

    logger.info("not implemented")

    # TODO: 
    # 1) learn what the best cutoff parameters are for both count-sketch and learned count sketch 
    # 2) run both 

     # learned algo with cutoff 
    test_results_learned_cutoff = []
    # if run_cutoff_experiment:
    #     logger.info("Running learned count sketch with cutoff")
    #     # learned algorithm with cutoff 
    #     pool = Pool(n_workers)
    #     results = pool.starmap(
    #         run_learned_count_sketch, zip(repeat(valid_data), repeat(valid_oracle_predictions), 
    #         space, space_fracs, partitions, repeat(True)))
    #     pool.close()
    #     pool.join()
        

    # vanilla sketch + cutoff 
    test_results_just_cutoff = []
    # if run_cutoff_experiment:
    #     logger.info("Running vanilla count sketch with cutoff")
    #     # count sketch algorithm with cutoff 
    #     pool = Pool(n_workers)
    #     results = pool.starmap(
    #         run_learned_count_sketch_just_cutoff, zip(repeat(data), repeat(oracle_predictions), 
    #         n_hashes, n_buckets)
    #     pool.close()
    #     pool.join()
    #     test_results_just_cutoff = [x[0] for x in results]
    #################################################################
    # save all results to the folder
    #################################################################
    # np.savez(os.path.join(save_folder, save_file),
    #     valid_count_sketch_predictions=test_count_sketch_predictions,
    #     test_count_sketch_predictions_cutoff=test_results_just_cutoff,
    #     n_hashes=n_hashes,
    #     space_list=space_list)

def experiment_comapre_loss_no_cutoff(
    valid_data,
    valid_oracle_scores,
    space_allocations,
    best_space_cs,
    best_space_cmin,
    best_partitions,
    n_workers, 
    save_folder, 
    save_file, 
    space_list,
    run_regular_count_sketch=False,
    run_learned_experiment=False, 
    run_perfect_oracle_experiment=False):

    # sort the data however we need it (here according to predicted scores)
    sort = np.argsort(valid_oracle_scores)[::-1]
    valid_oracle_predictions = valid_oracle_scores[sort]
    valid_data = valid_data[sort]
    true_values = valid_data

    #################################################################
    # evaluate vanilla sketching algorithm against learned sketches
    #################################################################

    # learned count sketch
    valid_algo_predictions = []
    loss_per_partition = []
    if run_learned_experiment:
        spinner = Halo(text='Evaluating learned count sketch', spinner='dots')
        spinner.start()

        with get_context("spawn").Pool() as pool:
            results = pool.starmap(
                run_learned_count_sketch, zip(repeat(valid_data), repeat(valid_oracle_predictions), 
                best_space_cs, best_space_cmin, best_partitions, repeat(False)))
            pool.close()
            pool.join()
      
        valid_algo_predictions = [x[0] for x in results]
        loss_per_partition = [x[1] for x in results]
        spinner.stop()

    # vanilla count sketch
    valid_count_sketch_predictions = []
    if run_regular_count_sketch:

        n_hashes = np.zeros(len(space_allocations), dtype=int)
        n_buckets = np.zeros(len(space_allocations), dtype=int)
        for i, space in enumerate(space_allocations):
            n_hashes[i] = 5
            n_buckets[i] = int(space/n_hashes[i])

        logger.info("Running vanilla count sketch")
        with get_context("spawn").Pool() as pool:
            test_count_sketch_predictions = pool.starmap(
                run_count_sketch, zip(repeat(valid_data), n_hashes, n_buckets))
            pool.close()
            pool.join()

    #################################################################
    # save all results to the folder
    #################################################################
    np.savez(os.path.join(save_folder, save_file),
        true_values=true_values,
        oracle_predictions=valid_oracle_predictions,
        valid_algo_predictions=valid_algo_predictions,
        valid_count_sketch_predictions=test_count_sketch_predictions,
        valid_loss_per_partition=loss_per_partition,
        space_list=space_list)

def order_y_wkey(y, results, key, n_examples=0):
    logger.info('loading results from %s' % results)
    results = np.load(results)
    pred_prob = results[key].squeeze()
    if n_examples:
        pred_prob = pred_prob[:n_examples]
    idx = np.argsort(pred_prob)[::-1]
    assert len(idx) == len(y)
    return y[idx], pred_prob[idx]

def order_y_wkey_list(y, results_list, key):
    pred_prob = np.array([])
    for results in results_list:
        results = np.load(results)
        pred_prob = np.concatenate((pred_prob, results[key].squeeze()))
    idx = np.argsort(pred_prob)[::-1]
    assert len(idx) == len(y)
    return y[idx], pred_prob[idx]

def load_dataset(dataset, model, key, is_aol=False, is_synth=False):

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

    spinner = Halo(text='Loading datasets...', spinner='dots')
    spinner.start()

    start_t = time.time()
    logger.info("Loading dataset...")
    if is_aol:
        _, y = get_data_aol_query_list(dataset)
    else:
        _, y = get_data_str_with_ports_list(dataset)
    
    logger.info('Done loading datasets (took %.1f sec)' % (time.time() - start_t))

    start_t = time.time()
    logger.info("Loading model...")
    data, oracle_scores = order_y_wkey_list(y, model, key)
    # if not is_aol:
    #     oracle_scores= np.exp(oracle_scores)
    logger.info('Done loading model (took %.1f sec)' % (time.time() - start_t))
    spinner.stop()

    sort = np.argsort(oracle_scores)[::-1]
    oracle_scores_sorted = oracle_scores[sort]
    data_sorted = data[sort]

    logger.info("///////////////////////////////////////////////////////////")
    logger.info("Dataset propertiess")
    logger.info("Size:        " + str(len(data_sorted)))
    logger.info("Data:        " + str(data_sorted))
    logger.info("Predictions: " + str(oracle_scores_sorted))
    logger.info("///////////////////////////////////////////////////////////")

    return data_sorted, oracle_scores_sorted


if __name__ == '__main__':
    set_start_method("spawn") # bug fix for deadlock in Pool: https://pythonspeed.com/articles/python-multiprocessing/

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--test_dataset", type=str, nargs='*', help="list of input .npy data for testing")
    argparser.add_argument("--valid_dataset", type=str, nargs='*', help="list of input .npy data for validation")
    argparser.add_argument("--optimal_params", type=str, nargs='*', help="optimal parameters to run the validation with")
    argparser.add_argument("--optimal_params_cutoff_cs", type=str, help="optimal parameters to run the validation with for cutoff count sketch")
    argparser.add_argument("--model", type=str, nargs='*', help="ml model to use as predictor (.npz file)")
    argparser.add_argument("--count_sketch_results", type=str,  help="results for count sketch .npz file")
    argparser.add_argument("--save_folder", type=str, required=True, help="folder to save the results in")
    argparser.add_argument("--save_file", type=str, required=True, help="prefix to save the results")
    argparser.add_argument("--seed", type=int, default=42, help="random state for sklearn")
    argparser.add_argument("--space_list", type=float, nargs='*', default=[], help="space in MB")
    argparser.add_argument("--n_workers", type=int, default=10, help="number of worker threads",)
    argparser.add_argument("--aol_data", action='store_true', default=False)
    argparser.add_argument("--synth_data", action='store_true', default=False)
    argparser.add_argument("--run_cutoff_version", action='store_true', default=False)
    argparser.add_argument("--run_learned_version", action='store_true', default=False)
    argparser.add_argument("--run_perfect_oracle_version", action='store_true', default=False)
    argparser.add_argument("--run_regular_count_sketch", action='store_true', default=False)
    args = argparser.parse_args()

    # set the random seed for numpy values
    np.random.seed(args.seed)
    
    # TODO: swap test and validation. Currently naming validation data when should be test data and vis versa

    if args.test_dataset is not None:
        # load the test dataset
        test_data, test_oracle_scores = load_dataset(
            args.test_dataset, 
            args.model, 
            'valid_output',
            args.aol_data,
            args.synth_data)
        
        # testing parameters 
        space_alloc = np.zeros(len(args.space_list))
        for i, space in enumerate(args.space_list):
            space_alloc[i] = int(space * 1e6 / 4)

        # find the best parameters for the algorithm on the test dataset 
        spinner = Halo(text='Finding optimal parameters for learned algorithm', spinner='dots')
        spinner.start()
        # TODO: do something about the appended file names; ideally have seperate optimal parameter files specified for cs and algo
        find_best_parameters_for_learned_algo(test_data, test_oracle_scores, args.space_list, space_alloc, args.n_workers, args.save_folder, args.save_file + '_learned', args.run_cutoff_version)
        spinner.stop()

        spinner = Halo(text='Finding optimal parameters for cutoff count sketch', spinner='dots')
        spinner.start()
        find_best_parameters_for_cutoff(test_data, test_oracle_scores, args.space_list, space_alloc, args.n_workers, args.save_folder,  args.save_file + '_count_sketch')
        spinner.stop()

    elif args.valid_dataset is not None:
        # load the test dataset
        valid_data, valid_oracle_scores = load_dataset(
            args.valid_dataset, 
            args.model, 
            'test_output',
            args.aol_data,
            args.synth_data)

        # figure out whether we need to load multiple param files
        best_cutoff_thresh_count_sketch = []
        best_cutoff_space_count_sketch = []
        learned_optimal_params = ''
        if len(args.optimal_params) > 1:
            learned_optimal_params = args.optimal_params[0]
            count_sketch_optimal_params = args.optimal_params[1]
            data = np.load(count_sketch_optimal_params)
            best_cutoff_space_count_sketch = np.array(data['best_space_cs'])
            best_cutoff_thresh_count_sketch = np.array(data['best_cutoff_thresh_for_space'])
        else:
            learned_optimal_params = args.optimal_params


        data = np.load(learned_optimal_params)
        space_list = np.array(data['space_list'])
        best_space_cs = np.array(data['best_space_cs'])
        best_space_cmin = np.array(data['best_space_cmin'])
        best_partitions = np.array(data['best_partitions_for_space'])
        space_allocations = np.array(data['space_allocations'])
            

        if args.run_cutoff_version:
            # run the experiment with the specified parameters
            experiment_comapre_loss_with_cutoff(
                valid_data, 
                valid_oracle_scores,
                space_allocations,
                best_space_cs,
                best_space_cmin,
                best_partitions,
                args.n_workers, 
                args.save_folder, 
                args.save_file, 
                space_list,
                run_regular_count_sketch=args.run_regular_count_sketch,
                run_learned_experiment=args.run_learned_version,
                run_perfect_oracle_experiment=args.run_perfect_oracle_version)
        else: 
            # run the experiment with the specified parameters
            experiment_comapre_loss_no_cutoff(
                valid_data, 
                valid_oracle_scores,
                space_allocations,
                best_space_cs,
                best_space_cmin,
                best_partitions,
                args.n_workers, 
                args.save_folder, 
                args.save_file, 
                space_list,
                run_regular_count_sketch=args.run_regular_count_sketch,
                run_learned_experiment=args.run_learned_version,
                run_perfect_oracle_experiment=args.run_perfect_oracle_version)

    else:
        logger.info("Error: need either testing or validation dataset")
        

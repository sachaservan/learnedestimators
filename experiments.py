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
logger.addHandler(logging.FileHandler('eval.log', 'a'))

#################################################
# each algorithm variant used in the experiemnts 
#################################################
def run_count_sketch(y, n_hashes, n_buckets):
    estimates = count_sketch(y, n_buckets, n_hashes)
    return estimates

def run_cutoff_count_sketch(y, y_scores, space, cutoff_threshold): 
    y_cutoff = y[cutoff_threshold:] # all items that have a predicted score > cutoff_thresh
    table_estimates = np.array(y[:cutoff_threshold]) # store exact counts for all 
    n_buckets = int(space / COUNT_MIN_OPTIMAL_N_HASH)
    n_hash = COUNT_SKETCH_OPTIMAL_N_HASH
    sketch_estimates = count_sketch(y_cutoff.copy(), n_buckets, n_hash)
    
    # prepend the table estimates to the count sketch estimates 
    all_estimates = table_estimates.tolist() + sketch_estimates.tolist()
    return all_estimates
  

def run_learned_count_sketch(y, y_scores, space_cs, space_cmin, partitions, cutoff=False): 

    if cutoff:
        # use all of space_cs for the cutoff table
        cutoff_threshold = int(space_cs / 2.0)
        y_cutoff = y[cutoff_threshold:] # all items that have a predicted score > cutoff_thresh
        y_scores_cutoff = y_scores[cutoff_threshold:]
        table_estimates = y[:cutoff_threshold] # store exact counts for all 
        sketch_estimates, loss_per_partition = learned_count_sketch_partitions(y_cutoff.copy(), y_scores_cutoff.copy(), 0, space_cmin, partitions)

        # prepend the table estimates to the count sketch estimates 
        all_estimates = table_estimates.tolist() + sketch_estimates.tolist()
        return all_estimates, loss_per_partition
    else:
        estimates, loss_per_partition = learned_count_sketch_partitions(y.copy(), y_scores.copy(), space_cs, space_cmin, partitions)
        return estimates, loss_per_partition

#################################################
def compute_partitions(scores, num_items_for_cs, n_partitions):
    splits = np.array_split(scores, n_partitions)
    sizes = [len(splits[k]) for k in range(n_partitions)]

    n_partitions_cs = 1
    n_items = sizes[0]
    for i in range(len(sizes)):
        if n_items + sizes[i] < num_items_for_cs:
            n_partitions_cs += 1
            n_items += sizes[i]
        else:
            break

    partitions = np.zeros(n_partitions - n_partitions_cs)
    for i in range(n_partitions_cs, n_partitions):   
        partitions[i - n_partitions_cs] = np.min(splits[i]) # min score as threshold 
    
    return partitions

def find_best_parameters_for_cutoff(test_data, test_oracle_scores, space_list, space_allocations, n_workers, save_folder, save_file):
    spinner = Halo(text='Finding optimal parameters for cutoff count sketch', spinner='dots')
    spinner.start()

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
            test_space_post_cutoff = int(test_space - test_space*test_cutoff_frac)
            test_space_cs.append(int(test_space_post_cutoff))

        logger.info("Learning best parameters for space setting...")
        start_t = time.time()

        test_cutoff_predictions = []
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

        spinner.info('Found optimal params for %.1f MB (took %.1f sec)' % (4*test_space/1e6, time.time() - start_t))
        spinner.start()
        logger.info('Found optimal params for %.1f MB (took %.1f sec)' % (4*test_space/1e6, time.time() - start_t))

        best_space_cs.append(space_cs)
        best_cutoff_thresh_for_space.append(cutoff_thresh)
        
    spinner.stop()
    np.savez(os.path.join(save_folder, save_file),
        space_list=space_list,
        best_cutoff_space_count_sketch=best_space_cs,
        best_cutoff_count_sketch_thresh_for_space=best_cutoff_thresh_for_space)

    return best_space_cs, best_cutoff_thresh_for_space

def find_best_parameters_for_learned_algo(test_data, test_oracle_scores, space_list, space_allocations, n_workers, save_folder, save_file, cutoff=False):
    space_fractions_to_test = SPACE_FRAC_TO_TEST
    n_partitions_to_test = NUM_PARTITIONS_TO_TEST

    if args.run_cutoff_version:
        spinner = Halo(text='Finding optimal parameters for learned algorithm with cutoff', spinner='dots')
    else:
        spinner = Halo(text='Finding optimal parameters for learned algorithm', spinner='dots')
    spinner.start()

    # compute partitions for each space allocation 
    best_space_cs = []
    best_space_cmin = []
    best_partitions_for_space = []
    best_cutoff_thresh_for_space = []
    for i, test_space in enumerate(space_allocations):
        test_space_cs = []
        test_space_cmin = []
        test_params_partitions = []

        # test all combinations 
        for space_frac in space_fractions_to_test:
            for test_n_partition in n_partitions_to_test:
                # combination of parameters to test
                test_space_cs.append(int(test_space * space_frac)/1.5) #TODO: deal with this magic constant 
                test_space_cmin.append(int(test_space * (1.0 - space_frac)))
                # TODO: figure out this constant; put in experiment_constants.py? 1.5 b/c otherwise worth storing in cutoff table
                num_items_for_cs = int(test_space * space_frac) 

                # avoid situation where everything is stored in count sketch...
                if num_items_for_cs >= len(test_data):
                    continue

                if args.run_cutoff_version: 
                    # in cutoff; we just take the space allocated for CS and use it to store items 
                    cutoff_thresh = int(test_space * space_frac / 2)
              
                    # avoid situation where everything is cut out 
                    if cutoff_thresh >= len(test_data)/2:
                        continue
                    
                    partitions = compute_partitions(test_oracle_scores[cutoff_thresh:], 0 , test_n_partition)
                    partitions = np.array([sys.maxsize] + partitions.tolist())
                    
                else:
                    partitions = compute_partitions(test_oracle_scores, num_items_for_cs , test_n_partition)
                  
                    # in case parameter combo leads to no space allocated to count sketch 
                    # need to ensure that all items are accounted for 
                    if num_items_for_cs == 0:
                        partitions = np.array([sys.maxsize] + partitions.tolist())

                test_params_partitions.append(partitions)

       
        spinner.info("Running " + str(len(test_space_cs)) + " different parameter combinations...")
        spinner.start()
       
        logger.info("Learning best parameters for space setting...")
       
        start_t = time.time()

        with get_context("spawn").Pool() as pool:
            results = pool.starmap(
                run_learned_count_sketch, 
                zip(repeat(test_data), 
                repeat(test_oracle_scores), 
                test_space_cs, 
                test_space_cmin, 
                test_params_partitions, 
                repeat(cutoff)))
            pool.close()
            pool.join()

        test_algo_predictions = [x[0] for x in results]
     
        losses = [np.sum(np.abs(test_data - predictions)**2) for predictions in test_algo_predictions]
        best_loss_idx = np.argmin(losses)
      
        space_cs = test_space_cs[best_loss_idx]
        space_cmin = test_space_cmin[best_loss_idx]
        partitions = test_params_partitions[best_loss_idx]
        best_space_cs.append(space_cs)
        best_space_cmin.append(space_cmin) # 1-space_frac (use all remaining space)
        best_partitions_for_space.append(partitions)


        spinner.info('Found optimal params for %.1f MB (took %.1f sec)' % (4*test_space/1e6, time.time() - start_t))
        spinner.start()

        logger.info('Found optimal params for %.1f MB (took %.1f sec)' % (4*test_space/1e6, time.time() - start_t))
        logger.info("All L2 losses:       " + str(losses))
        logger.info("Best L2 loss:        " + str(losses[best_loss_idx]))
        logger.info("Best space cs:       " + str(best_space_cs))
        logger.info("Best space cmin:     " + str(best_space_cmin))
        logger.info("Best partitions:     " + str(partitions))

    spinner.stop()
    np.savez(os.path.join(save_folder, save_file),
        space_list=space_list,
        best_space_cs=best_space_cs,
        best_space_cmin=best_space_cmin,
        best_partitions_for_space=best_partitions_for_space,
        space_allocations=space_allocations)

    return best_space_cs, best_space_cmin, best_partitions_for_space

def experiment_comapre_loss_with_cutoff(
    valid_data,
    valid_oracle_scores,
    space_alloc,
    best_space_cs,
    best_space_cmin,
    best_partitions,
    best_cutoff_cs_space,
    best_cutoff_cs_threshold,
    n_workers, 
    save_folder, 
    save_file, 
    space_list):

    # sort the data however we need it (here according to predicted scores)
    sort = np.argsort(valid_oracle_scores)[::-1]
    valid_oracle_predictions = valid_oracle_scores[sort]
    valid_data = valid_data[sort]
    true_values = valid_data

     # learned algo with cutoff 
    valid_cutoff_algo_predictions = []
    loss_per_partition = []
    logger.info("Running learned count sketch with cutoff")
    # learned algorithm with cutoff 
    with get_context("spawn").Pool() as pool:
        results = pool.starmap(
            run_learned_count_sketch, 
            zip(repeat(valid_data), 
            repeat(valid_oracle_scores), 
            best_space_cs, 
            best_space_cmin, 
            best_partitions, 
            repeat(True)))
        pool.close()
        pool.join()
    valid_cutoff_algo_predictions = [x[0] for x in results]
    loss_per_partition = [x[1] for x in results]

    # vanilla sketch + cutoff 
    valid_cutoff_count_sketch_predictions = []
    n_hashes = np.zeros(len(space_allocations), dtype=int)
    n_buckets = np.zeros(len(space_allocations), dtype=int)
    for i, space in enumerate(space_allocations):
        n_hashes[i] = 5
        n_buckets[i] = int(space/n_hashes[i])

    logger.info("Running vanilla count sketch")
    with get_context("spawn").Pool() as pool:
        valid_cutoff_count_sketch_predictions = pool.starmap(
            run_cutoff_count_sketch, 
            zip(repeat(valid_data), repeat(valid_oracle_scores), best_cutoff_cs_space, best_cutoff_cs_threshold))
        pool.close()
        pool.join()



    #################################################################
    # save all results to the folder
    #################################################################
    np.savez(os.path.join(save_folder, save_file),
        true_values=true_values,
        valid_cutoff_count_sketch_predictions=valid_cutoff_count_sketch_predictions,
        valid_cutoff_algo_predictions=valid_cutoff_algo_predictions,
        loss_per_partition=loss_per_partition,
        space_list=space_list)

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
    space_list):

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
    n_hashes = np.zeros(len(space_allocations), dtype=int)
    n_buckets = np.zeros(len(space_allocations), dtype=int)
    for i, space in enumerate(space_allocations):
        n_hashes[i] = 5
        n_buckets[i] = int(space/n_hashes[i])

    logger.info("Running regular count sketch")
    spinner = Halo(text='Evaluating regular count sketch', spinner='dots')
    spinner.start()
    with get_context("spawn").Pool() as pool:
        test_count_sketch_predictions = pool.starmap(
            run_count_sketch, zip(repeat(valid_data), n_hashes, n_buckets))
        pool.close()
        pool.join()
    spinner.stop()

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
    if not is_aol:
        oracle_scores = np.exp(oracle_scores)
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
    argparser.add_argument("--run_perfect_oracle_version", action='store_true', default=False)
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
            space_alloc[i] = int(space * 1e6 / 4.0) # 4 bytes per bucket

        # find the best parameters for the algorithm on the test dataset 
        # TODO: do something about the appended file names; ideally have seperate optimal parameter files specified for cs and algo
        find_best_parameters_for_learned_algo(test_data, test_oracle_scores, args.space_list, space_alloc, args.n_workers, args.save_folder, args.save_file + '_learned', args.run_cutoff_version)

        if args.run_cutoff_version:
            find_best_parameters_for_cutoff(test_data, test_oracle_scores, args.space_list, space_alloc, args.n_workers, args.save_folder,  args.save_file + '_count_sketch')

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
        learned_optimal_params = args.optimal_params[0]
        if len(args.optimal_params) > 1:
            count_sketch_optimal_params = args.optimal_params[1]
            data = np.load(count_sketch_optimal_params)
            best_cutoff_space_count_sketch = np.array(data['best_cutoff_space_count_sketch'])
            best_cutoff_thresh_count_sketch = np.array(data['best_cutoff_count_sketch_thresh_for_space'])

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
                best_cutoff_space_count_sketch,
                best_cutoff_thresh_count_sketch,
                args.n_workers, 
                args.save_folder, 
                args.save_file, 
                space_list)
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
                space_list)

    else:
        logger.info("Error: need either testing or validation dataset")
        

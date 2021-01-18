import os
import sys
import time
import csv
import argparse
import numpy as np

from halo import Halo
from itertools import repeat
from multiprocessing import Pool, set_start_method, get_context
from sketches import count_min, count_sketch, median_sketch
from learned_sketches import learned_count_sketch_partitions
from sketch_common import compute_distinct_element_sum_estimate, space_needed_for_distinct_element_sum_estiamte, estimate_median

from dataset import load_dataset

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
logger.addHandler(logging.FileHandler('experiments/eval.log', 'a'))

is_sorted = lambda a: np.all(a[:-1] <= a[1:])

##################################################$
# each algorithm variant used in the experiemnts #
###################################################
def run_count_sketch(y, space, sanity_check_space_bound):

    n_buckets = int(space / COUNT_SKETCH_OPTIMAL_N_HASH)
    n_hashes = COUNT_SKETCH_OPTIMAL_N_HASH
    estimates = count_sketch(y, n_buckets, n_hashes)

    if n_buckets * n_hashes > sanity_check_space_bound:
        print("[sanity check failed] TOTAL SPACE (vanilla count sketch) = " + str(n_buckets * n_hashes) + " > " + str(sanity_check_space_bound))
        exit(0)

    logger.info("Count Sketch space_used / space_allocated = " + str(n_buckets * n_hashes/ sanity_check_space_bound))


    return estimates

# 1) store cutoff_threshold items in a table and report their *exact* counts
# 2) store all items beyond the threshold in a count sketch and report the sketch counts
def run_cutoff_count_sketch(y, y_scores, space_for_sketch, cutoff_threshold, sanity_check_space_bound): 

    if not is_sorted(y_scores[::-1]):
        print("scores not sorted; is everything ok?")
        exit(0)

    y_cutoff = y[:cutoff_threshold]
    y_noncutoff = y[cutoff_threshold:] # all items that have a predicted score < cutoff_thresh
    table_estimates = np.array(y_cutoff) # store exact counts for all 

    sketch_estimates = run_count_sketch(y_noncutoff, space_for_sketch, space_for_sketch)

    # prepend the table estimates to the count sketch estimates 
    all_estimates = table_estimates.tolist() + sketch_estimates.tolist()

    space_used = space_for_sketch + cutoff_threshold * CUTOFF_SPACE_COST_FACTOR
    if space_used > sanity_check_space_bound:
        print("[sanity check failed] TOTAL SPACE (cutoff count sketch) = " + str(space_used) + " > " + str(sanity_check_space_bound))
        exit(0)

    logger.info("Cutoff Count Sketch space_used / space_allocated = " + str(space_used / sanity_check_space_bound))

    
    return all_estimates
  

# 1) store cutoff_threshold items in a table and report their *exact* counts
# 2) for all other items, report their (normalized) predicted frequency 
# use_exact_sum = True will not estimate the sum on the fly and instead compute the *exact* sum when normalizing predictions
# use_median_as_prediction = True will output the median of all non-cutoff values as the prediction for non-cutoff items 
def run_learned_predictions(y, y_scores, space, sanity_check_space_bound, use_exact_sum=False, use_median_as_prediction=False): 
    
    print("dataset size = " + str(len(y)))

    if not is_sorted(y_scores[::-1]):
        print("scores not sorted; is everything ok?")
        exit(0)
        
    # determine a bound on the space needed to compute a weighted sum estiamte 
    space_needed = space_needed_for_distinct_element_sum_estiamte(y_scores, DISTINCT_ELEMENT_SUM_ESTIMATE_ERROR, N_REGISTERS_FOR_HLL)

    # not enough space for a cutoff table, just output predictions 
    if space/CUTOFF_SPACE_COST_FACTOR <= space_needed:
        score_sum, _ = compute_distinct_element_sum_estimate(y_scores, DISTINCT_ELEMENT_SUM_ESTIMATE_ERROR, N_REGISTERS_FOR_HLL)
        pred_weights = y_scores / score_sum
        pred_estimates = np.sum(y) * pred_weights
        print("Not using cutoff table b/c " + str(space/CUTOFF_SPACE_COST_FACTOR) + "  < " + str(space_needed))
        return pred_estimates

    # allocate space for the cutoff table 
    # item | count (can just keep 4 byte "min score" so no need to store scores)
    table_cutoff = int((space - space_needed) / CUTOFF_SPACE_COST_FACTOR)
   
    # split into cutoff and noncutoff items 
    cutoff = y[:table_cutoff]
    noncutoff = y[table_cutoff:]

    # exact estimates for these items   
    cutoff_estimates = cutoff 

    space_used = table_cutoff*CUTOFF_SPACE_COST_FACTOR + space_needed
    if space_used > sanity_check_space_bound:
        print("[sanity check failed] TOTAL SPACE (learned count sketch) = " + str(space_used) + " > " + str(sanity_check_space_bound))
        exit(0)

    logger.info("Learned Sketch space_used / space_allocated = " + str(space_used / sanity_check_space_bound))


    # estimate the weights of oracle scores for non-cutoff items 
    noncutoff_scores = y_scores[table_cutoff:]
   
    noncutoff_score_sum = np.sum(noncutoff_scores)
    if not use_exact_sum:
        noncutoff_score_sum, _ = compute_distinct_element_sum_estimate(noncutoff_scores, DISTINCT_ELEMENT_SUM_ESTIMATE_ERROR, N_REGISTERS_FOR_HLL)
   
    pred_estimates = np.array([])
    if not use_median_as_prediction:
        # noncutoff_score_sum =  np.sum(noncutoff_scores)
        pred_weights = noncutoff_scores / noncutoff_score_sum

        # compute predictions for non-cutoff items 
        pred_estimates = np.sum(noncutoff) * pred_weights

    else:
        # output the median as the prediction for all non-cutoff values 
        pred_estimates = np.ones(len(noncutoff)) * np.median(noncutoff)

    # concat cutoff and non-cutoff predictions 
    all_estimates = cutoff_estimates.tolist() + pred_estimates.tolist()
    
    return all_estimates


def experiment_comapre_loss(
    algo_type,
    space_list,
    data,
    oracle_scores,
    space_allocations,
    best_cutoff_thresh_count_sketch,
    n_workers, 
    save_folder, 
    save_file):

    # learned algo with cutoff 
    logger.info("Running learned count sketch")
    
    algo_predictions = []
    algo_predictions_with_median = []
    cutoff_count_sketch_predictions = []
    count_sketch_prediction = []
    
    spinner = Halo(text='Evaluating learned predictions algorithm', spinner='dots')
    spinner.start()
    # special case; doesn't require loading optimal parameters
    # just run and exit 
    if algo_type == ALGO_TYPE_CUTOFF_AND_ORACLE:
          # learned algorithm with cutoff 
        with get_context("spawn").Pool() as pool:
            algo_predictions = pool.starmap(
                run_learned_predictions, 
                zip(repeat(data.copy()), 
                repeat(oracle_scores.copy()), 
                space_allocations,
                space_allocations,
                repeat(False),
                repeat(False)))
            pool.close()
            pool.join()

    if algo_type == ALGO_TYPE_CUTOFF_AND_ORACLE:
          # learned algorithm with cutoff 
        with get_context("spawn").Pool() as pool:
            algo_predictions_with_median = pool.starmap(
                run_learned_predictions, 
                zip(repeat(data.copy()), 
                repeat(oracle_scores.copy()), 
                space_allocations,
                space_allocations,
                repeat(False),
                repeat(True))) # run with median prediction 
            pool.close()
            pool.join()

    # vanilla sketch + cutoff 
    space_allocations_cutoff = []
    for i, space in enumerate(space_allocations):
        space_cutoff = space - best_cutoff_thresh_count_sketch[i] * CUTOFF_SPACE_COST_FACTOR # ID | count (4 bytes each)
        space_allocations_cutoff.append(space_cutoff)

    logger.info("Running cutoff count sketch on all parameters...")
    spinner.stop()
    spinner = Halo(text='Evaluating cutoff count sketch algorithm', spinner='dots')
    spinner.start()
    with get_context("spawn").Pool() as pool:
        cutoff_count_sketch_predictions = pool.starmap(
            run_cutoff_count_sketch, 
            zip(repeat(data.copy()), repeat(oracle_scores.copy()), space_allocations_cutoff, best_cutoff_thresh_count_sketch, space_allocations))
        pool.close()
        pool.join()

    # vanilla count sketch
    logger.info("Running vanilla count sketch on all parameters...")
    spinner.stop()
    spinner = Halo(text='Evaluating vanilla count sketch algorithm', spinner='dots')
    spinner.start()
    with get_context("spawn").Pool() as pool:
        count_sketch_predictions = pool.starmap(
            run_count_sketch, zip(repeat(data.copy()), space_allocations, space_allocations))
        pool.close()
        pool.join()

    spinner.stop()

    #################################################################
    # save all results to the folder
    #################################################################
    np.savez(os.path.join(save_folder, save_file),
        space_list=space_list,
        true_values=data,
        oracle_predictions=oracle_scores,
        algo_predictions=algo_predictions,
        algo_predictions_with_median=algo_predictions_with_median,
        cutoff_count_sketch_predictions=cutoff_count_sketch_predictions,
        count_sketch_predictions=count_sketch_predictions)


def experiment_comapre_loss_vs_oracle_error_on_synthetic_data(
    algo_type,
    space_list,
    data,
    space_allocations,
    n_workers, 
    save_folder, 
    save_file):

    # learned algo with cutoff 
    logger.info("Running learned count sketch on synthetic data with different prediction errors")
    
    oracle_scores = []
    algo_predictions_per_error = []
    true_counts_per_error = []
    algo_oracle_space_factor = []
   
    spinner = Halo(text='Evaluating learned predictions algorithm', spinner='dots')
    spinner.start()
    
    for space_factor in SYNTHETIC_DATA_ORACLE_SPACE_FACTOR_TO_TEST:
        # sample error
        scores = count_min(data, int(len(data)*space_factor), 1)
        scores /= 1000
        sort = np.argsort(scores)[::-1]
        data = data[sort]
        scores = scores[sort]

        # learned algorithm with cutoff 
        with get_context("spawn").Pool() as pool:
            algo_predictions = pool.starmap(
                run_learned_predictions, 
                zip(repeat(data.copy()), 
                repeat(scores), 
                space_allocations,
                space_allocations,
                repeat(False)))
            pool.close()
            pool.join()

        algo_predictions_per_error.append(algo_predictions)
        oracle_scores.append(scores)
        algo_oracle_space_factor.append(space_factor)
        true_counts_per_error.append(data) # keep track of sorted data for each error threshold


    # vanilla count sketch
    logger.info("Running vanilla count sketch on all parameters...")
    spinner.stop()
    spinner = Halo(text='Evaluating vanilla count sketch algorithm', spinner='dots')
    spinner.start()
    with get_context("spawn").Pool() as pool:
        count_sketch_predictions = pool.starmap(
            run_count_sketch, zip(repeat(data), space_allocations, space_allocations))
        pool.close()
        pool.join()

    spinner.stop()

    #################################################################
    # save all results to the folder
    #################################################################
    np.savez(os.path.join(save_folder, save_file),
        space_list=space_list,
        oracle_space_factor=algo_oracle_space_factor,
        true_values=data,
        true_counts_per_error=true_counts_per_error,
        oracle_scores=oracle_scores,
        algo_predictions_per_error=algo_predictions_per_error,
        count_sketch_predictions=count_sketch_predictions)


#################################################################
# optimal parameter finding for each algorithm
#################################################################

def find_best_parameters_for_cutoff(
    space_list,
    data, 
    oracle_scores, 
    space_allocations, 
    n_workers, 
    save_folder, 
    save_file):

    spinner = Halo(text='Finding optimal parameters for cutoff count sketch', spinner='dots')
    spinner.start()

    best_cutoff_thresh_for_space = []
    for i, test_space in enumerate(space_allocations):
        test_space_cs = []
        test_params_cutoff_thresh = []

        # test all combinations 
        for test_cutoff_frac in CUTOFF_FRAC_TO_TEST:
            # combination of parameters to test
            cutoff_thresh = int((test_cutoff_frac * test_space) / CUTOFF_SPACE_COST_FACTOR)
            test_params_cutoff_thresh.append(cutoff_thresh)
            test_space_post_cutoff = int(test_space - cutoff_thresh*CUTOFF_SPACE_COST_FACTOR)
            test_space_cs.append(int(test_space_post_cutoff))

        logger.info("Learning best parameters for space setting...")
        start_t = time.time()

        test_cutoff_predictions = []
        with get_context("spawn").Pool() as pool:
            test_cutoff_predictions = pool.starmap(
                run_cutoff_count_sketch, 
                zip(repeat(data), 
                repeat(oracle_scores), 
                test_space_cs, 
                test_params_cutoff_thresh, 
                repeat(test_space))
            )
            pool.close()
            pool.join()

      
        losses = [np.sum(np.abs(data - predictions)) for predictions in test_cutoff_predictions]
        best_loss_idx = np.argmin(losses)
        space_cs = test_space_cs[best_loss_idx]
        cutoff_thresh = test_params_cutoff_thresh[best_loss_idx]

        logger.info('Found optimal params for %.1f MB (took %.1f sec)' % (4*test_space/1e6, time.time() - start_t))

        best_cutoff_thresh_for_space.append(cutoff_thresh)
        
    spinner.stop()
    np.savez(os.path.join(save_folder, save_file),
        space_list=space_list,
        best_cutoff_thresh_for_space=best_cutoff_thresh_for_space)


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
    argparser.add_argument("--learned_algo_type", type=str, required=True, help="learned algorithm variant THRESHOLD | PARITION")
    argparser.add_argument("--synth_data", action='store_true', default=False)
    argparser.add_argument("--run_cutoff_count_sketch", action='store_true', default=False)
    argparser.add_argument("--run_perfect_oracle_version", action='store_true', default=False)
    argparser.add_argument("--model_size", type=float, default=0.0, help="model size in MB")
    args = argparser.parse_args()

    assert (args.learned_algo_type == ALGO_TYPE_CUTOFF_AND_ORACLE)

    # set the random seed for numpy values
    np.random.seed(args.seed)
    
    if args.test_dataset is not None:
        space_alloc = np.zeros(len(args.space_list))
        for i, space in enumerate(args.space_list):
            space_alloc[i] = int((space - args.model_size) * 1e6 / 4.0) # 4 bytes per bucket

        spinner = Halo(text='Loading datasets...', spinner='dots')
        spinner.start()

        # load the test dataset
        test_data, test_oracle_scores = load_dataset(
            args.test_dataset, 
            args.model, 
            'valid_output',
            args.run_perfect_oracle_version,
            args.aol_data,
            args.synth_data)
        spinner.stop()

        if args.run_cutoff_count_sketch:
            find_best_parameters_for_cutoff(
                args.space_list, 
                test_data, 
                test_oracle_scores, 
                space_alloc, 
                args.n_workers, 
                args.save_folder,  
                args.save_file + '_count_sketch')

        if args.learned_algo_type == ALGO_TYPE_CUTOFF_AND_ORACLE:
            print("This variant of the algorithm does not require test data")
            
            # save the space list used so that it can be loaded in the validation stage 
            np.savez(os.path.join(args.save_folder, args.save_file + '_learned'),
            best_cutoff_thresh_for_space=[],
            space_list=args.space_list)

    elif args.valid_dataset is not None:
        
        spinner = Halo(text='Loading datasets...', spinner='dots')
        spinner.start()

        # load the test dataset
        valid_data, valid_oracle_scores = load_dataset(
            args.valid_dataset, 
            args.model, 
            'test_output', # old models were trained with valid/test swapped 
            args.run_perfect_oracle_version,
            args.aol_data,
            args.synth_data)
        spinner.stop()


        median_est, space = estimate_median(valid_data[:10000])
        median_true = np.median(valid_data[:10000])

        print("estimated median: " + str(median_est))
        print("true median:      " + str(median_true))
        exit(0)


        # TODO: figure out whether we need to load multiple param files
        best_cutoff_thresh_count_sketch = []
        learned_optimal_params = args.optimal_params[0]
        if len(args.optimal_params) > 1:
            count_sketch_optimal_params = args.optimal_params[1]
            data = np.load(count_sketch_optimal_params)
            best_cutoff_thresh_count_sketch = np.array(data['best_cutoff_thresh_for_space'])

        data = np.load(learned_optimal_params)
        space_list = np.array(data['space_list'])

        space_alloc = np.zeros(len(space_list))
        for i, space in enumerate(space_list):
            space_alloc[i] = int((space - args.model_size) * 1e6 / 4.0) # 4 bytes per bucket

        if args.synth_data:
            # evaluate experiemnt with different errors for the oracle 
            experiment_comapre_loss_vs_oracle_error_on_synthetic_data(
                args.learned_algo_type,
                space_list,
                valid_data, 
                space_alloc,
                args.n_workers, 
                args.save_folder, 
                args.save_file + '_error_experiments')

            # run regular experiments (with cutoff) for comparisons 
            experiment_comapre_loss(
                args.learned_algo_type,
                space_list,
                valid_data, 
                valid_oracle_scores,
                space_alloc,
                best_cutoff_thresh_count_sketch,
                args.n_workers, 
                args.save_folder, 
                args.save_file)
        
        else:
            # run the experiment with the specified parameters
            experiment_comapre_loss(
                args.learned_algo_type,
                space_list,
                valid_data, 
                valid_oracle_scores,
                space_alloc,
                best_cutoff_thresh_count_sketch,
                args.n_workers, 
                args.save_folder, 
                args.save_file)
    else:
        logger.info("Error: need either testing or validation dataset")
        

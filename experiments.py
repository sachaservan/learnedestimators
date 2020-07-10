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
from learned_sketches import learned_count_sketch_partitions, learned_count_sketch_threshold
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

##################################################$
# each algorithm variant used in the experiemnts #
###################################################
def run_count_sketch(y, n_hashes, n_buckets):
    estimates = count_sketch(y, n_buckets, n_hashes)
    return estimates

# 1) store cutoff_threshold items in a table and report their *exact* counts
# 2) store all items beyond the threshold in a count sketch and report the sketch counts
def run_cutoff_count_sketch(y, y_scores, space, cutoff_threshold): 
   
    y_cutoff = y[cutoff_threshold:] # all items that have a predicted score > cutoff_thresh
    table_estimates = np.array(y[:cutoff_threshold]) # store exact counts for all 
   
    n_buckets = int(space / COUNT_MIN_OPTIMAL_N_HASH)
    n_hash = COUNT_SKETCH_OPTIMAL_N_HASH
    
    sketch_estimates = count_sketch(y_cutoff.copy(), n_buckets, n_hash)

    # prepend the table estimates to the count sketch estimates 
    all_estimates = table_estimates.tolist() + sketch_estimates.tolist()
    
    return all_estimates
  

# 1) store cutoff_threshold items in a table and report their *exact* counts
# 2) partition all remaining items according to the score partitions spceified
# 3) run count sketch within each partition and report a "corrected" count sketch count 
def run_learned_partition_count_sketch(y, y_scores, space, partitions, cutoff_threshold): 

    y_cutoff = y[cutoff_threshold:] # all items that have a predicted score > cutoff_thresh
    y_scores_cutoff = y_scores[cutoff_threshold:] # all items that have a predicted score > cutoff_thresh
    table_estimates = np.array(y[:cutoff_threshold]) # store exact counts for all 

    sketch_estimates, loss_per_partition = learned_count_sketch_partitions(y_cutoff.copy(), y_scores_cutoff.copy(), space, partitions)

    # prepend the table estimates to the count sketch estimates 
    all_estimates = table_estimates.tolist() + sketch_estimates.tolist()
    return all_estimates

# 1) store cutoff_threshold items in a table and report their *exact* counts
# 2) store all items beyond the threshold in a count sketch 
# 3) report count sketch count if it's outside the std_factor of the standard deviation 
#    from the normalized predicted score for the item 
def run_learned_threshold_count_sketch(y, y_scores, space, std_factor, cutoff_threshold): 
   
    y_cutoff = y[cutoff_threshold:] # all items that have a predicted score > cutoff_thresh
    y_scores_cutoff = y_scores[cutoff_threshold:] # all items that have a predicted score > cutoff_thresh
   
    table_estimates = np.array(y[:cutoff_threshold]) # store exact counts for all 
    sketch_estimates = learned_count_sketch_threshold(y_cutoff.copy(), y_scores_cutoff.copy(), space, std_factor)

    # prepend the table estimates to the count sketch estimates 
    all_estimates = table_estimates.tolist() + sketch_estimates.tolist()
    return all_estimates

# 1) store cutoff_threshold items in a table and report their *exact* counts
# 2) store all items beyond the cutoff_threshold in a count sketch provided they have a score
#    that is greater than sketch_score_threshold
# 3) all remaining items are counted (for normalization) but not stored anywhere and the normalized 
#    score counts are reported for them 
def run_learned_low_frequency_prediction_count_sketch(y, y_scores, space, cutoff_threshold, sketch_score_threshold): 
   
    y_cutoff = y[cutoff_threshold:] # all items that have a predicted score > cutoff_thresh
    y_scores_cutoff = y_scores[cutoff_threshold:] # all items that have a predicted score > cutoff_thresh
    table_estimates = np.array(y[:cutoff_threshold]) # store exact counts for all 

    y_sketch = y_cutoff[y_scores_cutoff > sketch_score_threshold] # all items that should be placed in the sketch
    y_scores_sketch = y_scores_cutoff[y_scores_cutoff > sketch_score_threshold]
  
    y_oracle = y_cutoff[y_scores_cutoff <= sketch_score_threshold] # all items that should be placed in the sketch
    y_scores_oracle = y_scores_cutoff[y_scores_cutoff <= sketch_score_threshold]

    n_hash = COUNT_SKETCH_OPTIMAL_N_HASH 
    n_buckets = int(space / n_hash)
    n_buckets -= int(EXTRA_SPACE_PER_PARTITION_IN_BYTES / (n_hash * 4)) # space is in units of 4 bytes 
    sketch_estimates = count_sketch(y_scores_sketch, n_buckets, n_hash)

    # TODO: use weighted distinct elements to normalize the scores
    # the results will be very close so this is fine for testing purposes 
    # (and we already deduct the space required for that with EXTRA_SPACE_PER_PARTITION_IN_BYTES) 
    lowfq_estimates = y_scores_oracle * np.sum(y_oracle) / np.sum(y_scores_oracle) 

    # prepend the table estimates to the count sketch estimates and the oracle estimates 
    all_estimates = table_estimates.tolist() + sketch_estimates.tolist() + lowfq_estimates.tolist()
    
    return all_estimates

# 1) store cutoff_threshold items in a table and report their *exact* counts
# 2) store all items beyond the threshold in a count sketch 
# 3) report count sketch count if it's outside the std_factor of the standard deviation 
#    from the normalized predicted score for the item 
def run_learned_cutoff_and_median(y, y_scores, space): 

    cutoff_threshold = int(space / 2.0)
   
    y_cutoff = y[cutoff_threshold:] # all items that have a predicted score > cutoff_thresh
    y_scores_cutoff = y_scores[cutoff_threshold:] # all items that have a predicted score > cutoff_thresh
   
    table_estimates = np.array(y[:cutoff_threshold]) # store exact counts for all 
    median_estiamtes = np.ones(len(y_cutoff)) * np.median(y_cutoff)

    # prepend the table estimates to the count sketch estimates 
    all_estimates = table_estimates.tolist() + median_estiamtes.tolist()
    return all_estimates


def experiment_comapre_loss(
    algo_type,
    space_list,
    data,
    oracle_scores,
    space_allocations,
    best_partitions,
    best_std_factors, 
    best_cutoff_thresholds_algo,
    best_sketch_thresholds_algo,
    best_cutoff_thresh_count_sketch,
    n_workers, 
    save_folder, 
    save_file):

    # learned algo with cutoff 
    logger.info("Running learned count sketch")
    
    algo_predictions = []
    if algo_type == ALGO_TYPE_PARTITION: 
        # learned algorithm with cutoff 
        with get_context("spawn").Pool() as pool:
            algo_predictions = pool.starmap(
                run_learned_partition_count_sketch, 
                zip(repeat(data), 
                repeat(oracle_scores), 
                space_allocations, 
                best_partitions, 
                best_cutoff_thresholds_algo))
            pool.close()
            pool.join()

    elif algo_type == ALGO_TYPE_STD_THRESHOLD: 
        # learned algorithm with cutoff 
        with get_context("spawn").Pool() as pool:
            algo_predictions = pool.starmap(
                run_learned_threshold_count_sketch, 
                zip(repeat(data), 
                repeat(oracle_scores), 
                space_allocations, 
                best_std_factors, 
                best_cutoff_thresholds_algo))
            pool.close()
            pool.join()

    elif algo_type == ALGO_TYPE_LOWFQ_PREDICTION: 
        # learned algorithm with cutoff 
        with get_context("spawn").Pool() as pool:
            algo_predictions = pool.starmap(
                run_learned_low_frequency_prediction_count_sketch, 
                zip(repeat(data), 
                repeat(oracle_scores), 
                space_allocations, 
                best_cutoff_thresholds_algo,
                best_sketch_threshold_algo))
            pool.close()
            pool.join()

    # special case; doesn't require loading optimal parameters
    # just run and exit 
    elif algo_type == ALGO_TYPE_CUTOFF_AND_MEDIAN:
          # learned algorithm with cutoff 
        with get_context("spawn").Pool() as pool:
            algo_predictions = pool.starmap(
                run_learned_cutoff_and_median, 
                zip(repeat(data), 
                repeat(oracle_scores), 
                space_allocations))
            pool.close()
            pool.join()
            

    # vanilla sketch + cutoff 
    n_hashes = np.zeros(len(space_allocations), dtype=int)
    n_buckets = np.zeros(len(space_allocations), dtype=int)
    for i, space in enumerate(space_allocations):
        n_hashes[i] = COUNT_SKETCH_OPTIMAL_N_HASH # TODO: change this to whatever the optimal really is
        n_buckets[i] = int(space/n_hashes[i])

    logger.info("Running cutoff count sketch on all parameters...")
    spinner = Halo(text='Evaluating cutoff count sketch', spinner='dots')
    with get_context("spawn").Pool() as pool:
        cutoff_count_sketch_predictions = pool.starmap(
            run_cutoff_count_sketch, 
            zip(repeat(data), repeat(oracle_scores), space_allocations, best_cutoff_thresh_count_sketch))
        pool.close()
        pool.join()


    # vanilla count sketch
    valid_count_sketch_predictions = []
    n_hashes = np.zeros(len(space_allocations), dtype=int)
    n_buckets = np.zeros(len(space_allocations), dtype=int)
    for i, space in enumerate(space_allocations):
        n_hashes[i] = COUNT_MIN_OPTIMAL_N_HASH
        n_buckets[i] = int(space/n_hashes[i])

    logger.info("Running regular count sketch on all parameters...")
    spinner.stop()
    spinner = Halo(text='Evaluating regular count sketch', spinner='dots')
    spinner.start()
    with get_context("spawn").Pool() as pool:
        count_sketch_predictions = pool.starmap(
            run_count_sketch, zip(repeat(data), n_hashes, n_buckets))
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
        cutoff_count_sketch_predictions=cutoff_count_sketch_predictions,
        count_sketch_predictions=count_sketch_predictions)



#################################################################
# optimal parameter finding for each algorithm
#################################################################

def compute_partitions(scores, n_partitions):
    splits = np.array_split(scores, n_partitions)
    sizes = [len(splits[k]) for k in range(n_partitions)]

    partitions = np.zeros(n_partitions)
    for i in range(n_partitions-1):   
        argmin = np.argmin(splits[i])
        partitions[i] = splits[i][argmin] # min score as threshold 

        # # move partition boundry such that there is no overlap in scores ... 
        # # important to avoid duplicates and such
        # j = 0
        # while partitions[i] == np.max(splits[i+1]):
        #     partitions[i] += splits[i][argmin-j]
        #     j += 1
        #     if argmin - j < 0:
        #         break 

    partitions[len(partitions) - 1] = sys.float_info.min 

    return partitions

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
            test_space_post_cutoff = int(test_space - test_space*test_cutoff_frac)
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
                test_params_cutoff_thresh))
            pool.close()
            pool.join()

      
        losses = [np.sum(np.abs(data - predictions)) for predictions in test_cutoff_predictions]
        best_loss_idx = np.argmin(losses)
        space_cs = test_space_cs[best_loss_idx]
        cutoff_thresh = test_params_cutoff_thresh[best_loss_idx]

        spinner.info('Found optimal params for %.1f MB (took %.1f sec)' % (4*test_space/1e6, time.time() - start_t))
        spinner.start()
        logger.info('Found optimal params for %.1f MB (took %.1f sec)' % (4*test_space/1e6, time.time() - start_t))

        best_cutoff_thresh_for_space.append(cutoff_thresh)
        
    spinner.stop()
    np.savez(os.path.join(save_folder, save_file),
        space_list=space_list,
        best_cutoff_thresh_for_space=best_cutoff_thresh_for_space)



def find_best_parameters_for_learned_partition_algo(
    space_list,
    data, 
    oracle_scores, 
    space_allocations, 
    n_workers, 
    save_folder, 
    save_file):


    spinner = Halo(text='Finding optimal parameters for learned algorithm', spinner='dots')
    spinner.start()

    # compute partitions for each space allocation 
    best_partitions_for_space = []
    best_cutoff_thresh_for_space = []
 
    for i, test_space in enumerate(space_allocations):
        test_params_partitions = []
        test_params_cutoff_thresh = []
        test_params_space = []

        # test all combinations 
        for test_cutoff_frac in CUTOFF_FRAC_TO_TEST:
            for test_n_partition in NUM_PARTITIONS_TO_TEST:
                # combination of parameters to test
                partitions = compute_partitions(oracle_scores, test_n_partition)
                cutoff_thresh = int((test_cutoff_frac * test_space) / CUTOFF_SPACE_COST_FACTOR)
               
                test_params_partitions.append(partitions)
                test_space_post_cutoff = int(test_space - test_space*test_cutoff_frac)
                test_params_cutoff_thresh.append(cutoff_thresh)
                test_params_space.append(int(test_space_post_cutoff))

        spinner.info("Running " + str(len(test_params_space)) + " different parameter combinations...")
        spinner.start()
        logger.info("Learning best parameters for space setting...")
        
        start_t = time.time()

        with get_context("spawn").Pool() as pool:
            test_algo_predictions = pool.starmap(
                run_learned_partition_count_sketch, 
                zip(repeat(data), 
                repeat(oracle_scores), 
                test_params_space, 
                test_params_partitions,
                test_params_cutoff_thresh))
            pool.close()
            pool.join()
     
        losses = [np.sum(np.abs(data - predictions)) for predictions in test_algo_predictions]
        best_loss_idx = np.argmin(losses)
      
        partitions = test_params_partitions[best_loss_idx]
        cutoff_thresh = test_params_cutoff_thresh[best_loss_idx]
        best_partitions_for_space.append(partitions)
        best_cutoff_thresh_for_space.append(cutoff_thresh)

        spinner.info('Found optimal params for %.1f MB (took %.1f sec)' % (4*test_space/1e6, time.time() - start_t))
        spinner.start()

        logger.info('Found optimal params for %.1f MB (took %.1f sec)' % (4*test_space/1e6, time.time() - start_t))
        logger.info("All L1 losses:       " + str(losses))
        logger.info("Best L1 loss:        " + str(losses[best_loss_idx]))
        logger.info("Best partitions:     " + str(partitions))

    spinner.stop()
    np.savez(os.path.join(save_folder, save_file),
        space_list=space_list,
        best_partitions_for_space=best_partitions_for_space,
        best_cutoff_thresh_for_space=best_cutoff_thresh_for_space)

def find_best_parameters_for_learned_threshold_algo(
    space_list,
    data, 
    oracle_scores, 
    space_allocations, 
    n_workers, 
    save_folder, 
    save_file):

    spinner = Halo(text='Finding optimal parameters for learned algorithm', spinner='dots')
    spinner.start()

    # compute partitions for each space allocation 
    best_std_factor_for_space = []
    best_cutoff_thresh_for_space = []
 
    for i, test_space in enumerate(space_allocations):
        test_params_std_factors = []
        test_params_cutoff_thresh = []
        test_params_space = []

        # test all combinations 
        for test_cutoff_frac in CUTOFF_FRAC_TO_TEST:
            for std_factor in STD_FACTORS_TO_TEST:
                # combination of parameters to test
                cutoff_thresh = int((test_cutoff_frac * test_space) / CUTOFF_SPACE_COST_FACTOR)
               
                test_params_std_factors.append(std_factor)
                test_space_post_cutoff = int(test_space - test_space*test_cutoff_frac)
                test_params_cutoff_thresh.append(cutoff_thresh)
                test_params_space.append(int(test_space_post_cutoff))

        spinner.info("Running " + str(len(test_params_space)) + " different parameter combinations...")
        spinner.start()
        logger.info("Learning best parameters for space setting...")
        
        start_t = time.time()

        with get_context("spawn").Pool() as pool:
            test_algo_predictions = pool.starmap(
                run_learned_threshold_count_sketch, 
                zip(repeat(data), 
                repeat(oracle_scores), 
                test_params_space, 
                test_params_std_factors,
                test_params_cutoff_thresh))
            pool.close()
            pool.join()
     
        losses = [np.sum(np.abs(data - predictions)) for predictions in test_algo_predictions]
        best_loss_idx = np.argmin(losses)
      
        std_factor = test_params_std_factors[best_loss_idx]
        cutoff_thresh = test_params_cutoff_thresh[best_loss_idx]
        best_std_factor_for_space.append(std_factor)
        best_cutoff_thresh_for_space.append(cutoff_thresh)

        spinner.info('Found optimal params for %.1f MB (took %.1f sec)' % (4*test_space/1e6, time.time() - start_t))
        spinner.start()

        logger.info('Found optimal params for %.1f MB (took %.1f sec)' % (4*test_space/1e6, time.time() - start_t))
        logger.info("All L1 losses:       " + str(losses))
        logger.info("Best L1 loss:        " + str(losses[best_loss_idx]))
        logger.info("Best std factors:     " + str(std_factor))

    spinner.stop()
    np.savez(os.path.join(save_folder, save_file),
        space_list=space_list,
        best_std_factors_for_space=best_std_factor_for_space,
        best_cutoff_thresh_for_space=best_cutoff_thresh_for_space)


def find_best_parameters_for_learned_lowfq_prediction_algo(
    space_list,
    data, 
    oracle_scores, 
    space_allocations, 
    n_workers, 
    save_folder, 
    save_file):

    spinner = Halo(text='Finding optimal parameters for learned algorithm', spinner='dots')
    spinner.start()

    # compute partitions for each space allocation 
    best_sketch_thresh_for_space = []
    best_cutoff_thresh_for_space = []
 
    for i, test_space in enumerate(space_allocations):
        test_params_sketch_thresh = []
        test_params_cutoff_thresh = []
        test_params_space = []

        # test all combinations 
        for test_cutoff_frac in CUTOFF_FRAC_TO_TEST:
            for test_sketch_frac in SKETCH_FRAC_TO_TEST:
                # combination of parameters to test
                cutoff_thresh = int((test_cutoff_frac * test_space) / CUTOFF_SPACE_COST_FACTOR)

                # fraction of items that should be in either cutoff or sketch
                sketch_thresh_index = int(test_sketch_frac * len(oracle_scores))
                sketch_thresh = oracle_scores[sketch_thresh_index]

                test_params_sketch_thresh.append(sketch_thresh)
                test_space_post_cutoff = int(test_space - test_space*test_cutoff_frac)
                test_params_cutoff_thresh.append(cutoff_thresh)
                test_params_space.append(int(test_space_post_cutoff))

        spinner.info("Running " + str(len(test_params_space)) + " different parameter combinations...")
        spinner.start()
        logger.info("Learning best parameters for space setting...")
        
        start_t = time.time()

        with get_context("spawn").Pool() as pool:
            test_algo_predictions = pool.starmap(
                run_learned_low_frequency_prediction_count_sketch, 
                zip(repeat(data), 
                repeat(oracle_scores), 
                test_params_space, 
                test_params_cutoff_thresh,
                test_params_sketch_thresh))
            pool.close()
            pool.join()
     
        losses = [np.sum(np.abs(data - predictions)) for predictions in test_algo_predictions]
        best_loss_idx = np.argmin(losses)
      
        sketch_thresh = test_params_sketch_thresh[best_loss_idx]
        cutoff_thresh = test_params_cutoff_thresh[best_loss_idx]
        best_sketch_thresh_for_space.append(sketch_thresh)
        best_cutoff_thresh_for_space.append(cutoff_thresh)

        spinner.info('Found optimal params for %.1f MB (took %.1f sec)' % (4*test_space/1e6, time.time() - start_t))
        spinner.start()

        logger.info('Found optimal params for %.1f MB (took %.1f sec)' % (4*test_space/1e6, time.time() - start_t))
        logger.info("All L1 losses:       " + str(losses))
        logger.info("Best L1 loss:        " + str(losses[best_loss_idx]))
        logger.info("Best sketch thresh:  " + str(sketch_thresh))

    spinner.stop()
    np.savez(os.path.join(save_folder, save_file),
        space_list=space_list,
        best_sketch_thresh_for_space=best_sketch_thresh_for_space,
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

    assert (args.learned_algo_type == ALGO_TYPE_PARTITION \
        or args.learned_algo_type == ALGO_TYPE_STD_THRESHOLD \
        or args.learned_algo_type == ALGO_TYPE_LOWFQ_PREDICTION \
        or args.learned_algo_type == ALGO_TYPE_CUTOFF_AND_MEDIAN)

    # set the random seed for numpy values
    np.random.seed(args.seed)
    
    if args.valid_dataset is not None:

        space_alloc = np.zeros(len(args.space_list))
        for i, space in enumerate(args.space_list):
            space_alloc[i] = int((space - args.model_size) * 1e6 / 4.0) # 4 bytes per bucket

        spinner = Halo(text='Loading datasets...', spinner='dots')
        spinner.start()

        # load the test dataset
        valid_data, valid_oracle_scores = load_dataset(
            args.valid_dataset, 
            args.model, 
            'valid_output',
            args.run_perfect_oracle_version,
            args.aol_data,
            args.synth_data)
        spinner.stop()

        # TODO: make these constants 
        if args.learned_algo_type == ALGO_TYPE_PARTITION:
            find_best_parameters_for_learned_partition_algo(
                args.space_list, 
                valid_data, 
                valid_oracle_scores, 
                space_alloc, 
                args.n_workers, 
                args.save_folder, 
                args.save_file + '_learned')

        elif args.learned_algo_type == ALGO_TYPE_STD_THRESHOLD:
            find_best_parameters_for_learned_threshold_algo(
                args.space_list, 
                valid_data, 
                valid_oracle_scores, 
                space_alloc, 
                args.n_workers, 
                args.save_folder, 
                args.save_file + '_learned')
      
        elif args.learned_algo_type == ALGO_TYPE_LOWFQ_PREDICTION:
            find_best_parameters_for_learned_lowfq_prediction_algo(
                args.space_list, 
                valid_data, 
                valid_oracle_scores, 
                space_alloc, 
                args.n_workers, 
                args.save_folder, 
                args.save_file + '_learned')
        elif args.learned_algo_type == ALGO_TYPE_CUTOFF_AND_MEDIAN:
            print("This variant of the algorithm does not require validation data")
            exit(0)



        if args.run_cutoff_count_sketch:
            find_best_parameters_for_cutoff(
                args.space_list, 
                valid_data, 
                valid_oracle_scores, 
                space_alloc, 
                args.n_workers, 
                args.save_folder,  
                args.save_file + '_count_sketch')



    elif args.test_dataset is not None:
        
        spinner = Halo(text='Loading datasets...', spinner='dots')
        spinner.start()

        # load the test dataset
        test_data, test_oracle_scores = load_dataset(
            args.test_dataset, 
            args.model, 
            'test_output',
            args.run_perfect_oracle_version,
            args.aol_data,
            args.synth_data)
        spinner.stop()


        # TODO: figure out whether we need to load multiple param files
        best_cutoff_thresh_count_sketch = []
        best_cutoff_space_count_sketch = []
        learned_optimal_params = args.optimal_params[0]
        if len(args.optimal_params) > 1:
            count_sketch_optimal_params = args.optimal_params[1]
            data = np.load(count_sketch_optimal_params)
            best_cutoff_thresh_count_sketch = np.array(data['best_cutoff_thresh_for_space'])

        data = np.load(learned_optimal_params)
        space_list = np.array(data['space_list'])


        # select optimal parameters based on what algo version being run 
        best_partitions_algo = []
        best_std_factors_algo = [] 
        best_sketch_threshold_algo = [] 

        if args.learned_algo_type == ALGO_TYPE_PARTITION:
            best_partitions_algo = np.array(data['best_partitions_for_space']) 
        elif args.learned_algo_type == ALGO_TYPE_STD_THRESHOLD:
            best_std_factors_algo = np.array(data['best_std_factors_for_space'])
        elif args.learned_algo_type == ALGO_TYPE_LOWFQ_PREDICTION:
            best_sketch_threshold_algo = np.array(data['best_sketch_thresh_for_space'])
      
        # elif args.learned_algo_type == ALGO_TYPE_CUTOFF_AND_MEDIAN:
        # no parameters to load for this algorithm 

        # load the best cutoff threshold for the algorithm (present in all optimal param files)
        best_cutoff_thresh_algo = np.array(data['best_cutoff_thresh_for_space'])

        space_alloc = np.zeros(len(space_list))
        for i, space in enumerate(space_list):
            space_alloc[i] = int((space - args.model_size) * 1e6 / 4.0) # 4 bytes per bucket

        # run the experiment with the specified parameters
        experiment_comapre_loss(
            args.learned_algo_type,
            space_list,
            test_data, 
            test_oracle_scores,
            space_alloc,
            best_partitions_algo,
            best_std_factors_algo,
            best_cutoff_thresh_algo,
            best_sketch_threshold_algo,
            best_cutoff_thresh_count_sketch,
            args.n_workers, 
            args.save_folder, 
            args.save_file)
    else:
        logger.info("Error: need either testing or validation dataset")
        

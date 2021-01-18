import os
import sys
import time
import csv
import argparse
import numpy as np

from itertools import repeat
from multiprocessing import Pool, set_start_method, get_context
from sketches import count_min, count_sketch, median_sketch
from learned_sketches import learned_count_sketch_partitions
from experiment_constants import *

# utils for parsing the dataset and results 
from utils import get_data_aol_query_list
from utils import get_data, get_stat, git_log, feat_to_string, get_data_str_with_ports_list

# setup logging 
import logging
logger = logging.getLogger('learned_estimators_log')
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger.propagate = False
logger.addHandler(logging.FileHandler('experiments/eval.log', 'a'))


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

def load_dataset(dataset, model, key, perfect_oracle=False, is_aol=False, is_synth=False):

    if is_synth:
        N = 250000
        a = 2 # zipf param
        data = np.random.zipf(a, N).flatten() + np.ones(N)
        sort = np.argsort(data)[::-1]
        data = data[sort]
        data = data * 1000
        scores = count_sketch(data, int(len(data)*0.3), 3)
        sort = np.argsort(scores)[::-1]
        data = data[sort]
        scores = scores[sort]

        return data, scores

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

    # IP data is stored in log form
    if not is_aol:
        oracle_scores = np.exp(oracle_scores)
    logger.info('Done loading model (took %.1f sec)' % (time.time() - start_t))

    logger.info("///////////////////////////////////////////////////////////")
    logger.info("Dataset propertiess")
    logger.info("Size:        " + str(len(data)))
    logger.info("Data:        " + str(data))
    logger.info("Predictions: " + str(oracle_scores))
    logger.info("///////////////////////////////////////////////////////////")

    if perfect_oracle:
        oracle_scores = data / np.sum(data) # perfect predictions 
        sort = np.argsort(oracle_scores)[::-1]
        oracle_scores = oracle_scores[sort]
        data = data[sort]

    return data, oracle_scores



###############################################################
# TODO: move stat code, etc. into experiment file 
###############################################################
def stats_similarity_between_top_valid_and_test(valid_x, valid_y, test_x, test_y, top_frac=0.05):

    # sort valid and test data by frequency 
    sort_valid = np.argsort(valid_y)[::-1]
    sort_test = np.argsort(test_y)[::-1]

    valid_x = valid_x[sort_valid]
    valid_y = valid_y[sort_valid]

    test_x = test_x[sort_test]
    test_y = test_y[sort_test]

    # TODO: find a faster way of computing features in common; really slow right now 

    N = int(len(sort_valid)*top_frac)
    common = 0
    for i in range(N):
        for j in range(N):
            if valid_x[i] == test_x[j]:
                common += 1
                break
    print("[stats] number of features in common: {} for top-{}\%".format(common / N, top_frac*100))


def stats_median_prediction(scores_y, test_y):

    median_pred_y_02 = median_sketch(test_y, 16666 * 3)
    median_pred_y_05 = median_sketch(test_y, 41666 * 3)

    learned_median_pred_y_02 = learned_median_sketch(test_y, scores_y, 16666 * 3)
    learned_median_pred_y_05 = learned_median_sketch(test_y, scores_y, 41666 * 3)

    pred_y = scores_y * np.sum(scores_y) / np.sum(test_y) # normalize the oracle scores 
    N = np.sum(test_y)

    print("[stats] sketch median (0.2MB) " + str(median_pred_y_02) + "actual median: " + str(np.median(test_y)))
    print("[stats] sketch median (0.5MB) " + str(median_pred_y_05) + "actual median: " + str(np.median(test_y)))
    print("[stats] learned sketch median (0.2MB) " + str(learned_median_pred_y_02) + "actual median: " + str(np.median(test_y)))
    print("[stats] learned sketch median (0.5MB) " + str(learned_median_pred_y_05) + "actual median: " + str(np.median(test_y)))




def stats_oracle_predictions_valid_and_test(valid_x, valid_y, valid_scores, test_x, test_y, test_scores, top_frac=0.05):

    # normalize the scores 
    valid_scores =  valid_scores * np.sum(valid_y) / np.sum(valid_scores)
    test_scores =  test_scores * np.sum(test_y) / np.sum(test_scores)

    # sort valid and test data by frequency 
    sort_valid = np.argsort(valid_y)[::-1]
    sort_test = np.argsort(test_y)[::-1]

    valid_x = valid_x[sort_valid]
    valid_y = valid_y[sort_valid]
    valid_scores = valid_scores[sort_valid]

    test_x = test_x[sort_test]
    test_y = test_y[sort_test]
    test_scores = test_scores[sort_test]

    N = int(len(sort_valid)*top_frac)
    print("[stats] avg. loss in oracle prediction on valid: {} for top-{}\%".format(np.sum(np.abs(valid_y[:N] - valid_scores[:N])) / N, top_frac*100))
    print("[stats] avg. loss in oracle prediction on test: {} for top-{}\%".format(np.sum(np.abs(test_y[:N] - test_scores[:N])) / N, top_frac*100))



def stats_just_oracle_predictions(scores_y, test_y):

    cs_pred_y_02 = count_sketch(test_y, 16666, 3)
    cs_pred_y_05 = count_sketch(test_y, 41666, 3)

    pred_y = scores_y * np.sum(scores_y) / np.sum(test_y) # normalize the oracle scores 
    N = np.sum(test_y)

    print("[stats] cs (0.2MB) loss Abs: " + str(np.sum(np.abs(cs_pred_y_02 - test_y))/N))
    print("[stats] cs (0.2MB) loss MSE: " + str(np.sum(np.abs(cs_pred_y_02 - test_y)**2)/N))
    print("[stats] cs (0.2MB) loss Weighted: " + str(np.sum(np.abs(cs_pred_y_02 - test_y)*test_y)/N))
    print("[stats] cs (0.2MB) loss Relative: " + str(np.sum(np.abs(cs_pred_y_02 - test_y)/test_y)/N))

    print("[stats] cs (0.5MB) loss Abs: " + str(np.sum(np.abs(cs_pred_y_05 - test_y))/N))
    print("[stats] cs (0.5MB) loss MSE: " + str(np.sum(np.abs(cs_pred_y_05 - test_y)**2)/N))
    print("[stats] cs (0.5MB) loss Weighted: " + str(np.sum(np.abs(cs_pred_y_05 - test_y)*test_y)/N))
    print("[stats] cs (0.5MB) loss Relative: " + str(np.sum(np.abs(cs_pred_y_05 - test_y)/test_y)/N))


    print("[stats] model loss Abs: " + str(np.sum(np.abs(pred_y - test_y))/N))
    print("[stats] model loss MSE: " + str(np.sum(np.abs(pred_y - test_y)**2)/N))
    print("[stats] model loss Weighted: " + str(np.sum(np.abs(pred_y - test_y)*test_y)/N))
    print("[stats] model loss Relative: " + str(np.sum(np.abs(pred_y - test_y)/test_y)/N))




if __name__ == '__main__':
    set_start_method("spawn") # bug fix for deadlock in Pool: https://pythonspeed.com/articles/python-multiprocessing/

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--test_dataset", type=str, nargs='*', help="list of input .npy data for testing")
    argparser.add_argument("--valid_dataset", type=str, nargs='*', help="list of input .npy data for validation")
    argparser.add_argument("--model_valid", type=str, nargs='*', help="ml model to use as predictor (.npz file)")
    argparser.add_argument("--model_test", type=str, nargs='*', help="ml model to use as predictor (.npz file)")
    argparser.add_argument("--seed", type=int, default=42, help="random state for sklearn")
    argparser.add_argument("--frac", type=float, default=0.1, help="fraction of top-items to compare")
    argparser.add_argument("--aol_data", action='store_true', default=False)
    argparser.add_argument("--synth_data", action='store_true', default=False)
    args = argparser.parse_args()

    if args.aol_data:
        valid_x, valid_y = get_data_aol_query_list(args.valid_dataset)
    else:
        valid_x, valid_y = get_data_str_with_ports_list(args.valid_dataset)

    valid_y, valid_scores = order_y_wkey_list(valid_y, args.model_valid, 'valid_output')

    if args.aol_data:
        test_x, test_y = get_data_aol_query_list(args.test_dataset)
    else:
        test_x, test_y = get_data_str_with_ports_list(args.test_dataset)

    test_y, test_scores = order_y_wkey_list(test_y, args.model_test, 'test_output')


    # compare median with predicted median 
    stats_median_prediction(valid_scores, valid_y)
    exit(0)
    
    # compares (normalized) oracle prediction accuracy between test and valid data 
    # stats_oracle_predictions_valid_and_test(valid_x, valid_y, valid_scores, test_x, test_y, test_scores, 0.01)
    # stats_oracle_predictions_valid_and_test(valid_x, valid_y, valid_scores, test_x, test_y, test_scores, 0.05)
    # stats_oracle_predictions_valid_and_test(valid_x, valid_y, valid_scores, test_x, test_y, test_scores, 0.1)

    # compares the number of features in common between valid and test data 
    # stats_similarity_between_top_valid_and_test(valid_x, valid_y, test_x, test_y, 0.01)
    # stats_similarity_between_top_valid_and_test(valid_x, valid_y, test_x, test_y, 0.05)
    # stats_similarity_between_top_valid_and_test(valid_x, valid_y, test_x, test_y, 0.1)

    # stats_just_oracle_predictions(test_scores, test_y)
   # stats_number_of_features_in_common(valid_x, test_x)


    test_y_total = np.sum(test_y)

    valid_n = int(args.frac * len(valid_x))
    valid_x = valid_x[:valid_n]
    valid_y = valid_y[:valid_n]
    valid_scores = valid_scores[:valid_n]


    # compute count sketch loss for high frequency and low frequency items 
    pred_y_cs_freq = count_sketch(test_y, 16666, 3)
    test_cs_loss = np.sum(np.abs(pred_y_cs_freq - test_y) / test_y)

    test_n = int(args.frac * len(test_x))
    pred_y_low_freq = count_sketch(test_y[:test_n], 16666, 3)
    abs_diff_low_freq = np.abs(pred_y_low_freq - test_y[:test_n])
   

    cutoff_n = 1000
    pred_y_cutoff_cs_freq = count_sketch(test_y[cutoff_n:], 16666 - int(cutoff_n*(4/3)), 3)
    pred_y_cutoff_cs_freq = np.concatenate((test_y[:cutoff_n], pred_y_cutoff_cs_freq))
    test_cutoff_cs_loss = np.sum(np.abs(pred_y_cutoff_cs_freq - test_y) / test_y)

    sum_y_high_freq = np.sum(test_y[:test_n])
    high_freq_scores = test_scores[test_n:]
    pred_test =  (high_freq_scores /  np.sum(high_freq_scores)) * np.sum(test_y[test_n:])

    #print("test loss high frequency " + str(np.sum(np.abs(pred_test - test_y[:test_n])*test_y[:test_n]) / sum_y_high_freq))

    test_loss_abs = np.abs(pred_test - test_y[test_n:])
    test_loss_weighted = np.abs(pred_test - test_y[test_n:])/test_y[test_n:]
    test_loss_abs = test_loss_abs.astype(int)
    test_loss_weighted = test_loss_weighted.astype(int)
    


    print("high frequency loss (relative weighted):")
    print(test_loss_weighted)

    print("cs loss (relative weighted):")
    print(test_cs_loss)

    print("total loss cutoff (relative weighted):")
    print(test_cutoff_cs_loss)

    print("total loss (relative weighted):")
    print(np.sum(test_loss_weighted) + np.sum(abs_diff_low_freq / test_y[:test_n]))

    model_acc = []


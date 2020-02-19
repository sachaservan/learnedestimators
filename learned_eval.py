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
from learned_sketches import dynamic_count_min, dynamic_count_sketch, test_oracle_count_sketch, test_oracle_count_min_sketch
from aol_utils import get_data_aol_query_list

def loss_weighted(y_true, y_est):
    return np.abs(y_true - y_est) * y_true

def loss_l1(y_true, y_est):
    return np.abs(y_true - y_est)

def loss_l2(y_true, y_est):
    return np.abs(y_true - y_est) ** 2

loss_function = loss_weighted

def run_countminsketch(y, n_hashes, n_buckets, name):
    loss = count_min(y, n_buckets, n_hashes, loss_function)
    print('%s: # hashes %d, # buckets %d - loss %.2f\t time: %.2f sec' % \
        (name, n_hashes, n_buckets, loss, time.time() - start_t))
    return loss


def run_countsketch(y, n_hashes, n_buckets, name):
    loss = count_sketch(y, n_buckets, n_hashes, loss_function)
    print('%s: # hashes %d, # buckets %d - loss %.2f\t time: %.2f sec' % \
        (name, n_hashes, n_buckets, loss, time.time() - start_t))
    return loss


def run_dynamicminsketch(y, y_scores, n_hashes, n_buckets, name):
    start_t = time.time()
    loss = dynamic_count_min(y, y_scores, n_hashes, n_buckets, loss_function)
    print('%s: # hashes %d, # buckets %d - loss %.2f\t time: %.2f sec' % \
        (name, n_hashes, n_buckets, loss, time.time() - start_t))
    return loss

def run_dynamicsketch(y, y_scores, n_hashes, n_buckets, name):
    start_t = time.time()
    loss = dynamic_count_sketch(y, y_scores, n_hashes, n_buckets, loss_function)
    print('%s: # hashes %d, # buckets %d - loss %.2f\t time: %.2f sec' % \
        (name, n_hashes, n_buckets, loss, time.time() - start_t))
    return loss

def run_testsketch(y, y_scores, n_hashes, n_buckets, name):
    start_t = time.time()
    loss = test_oracle_count_sketch(y, y_scores, n_hashes, n_buckets, loss_function)
    print('%s: # hashes %d, # buckets %d - loss %.2f\t time: %.2f sec' % \
        (name, n_hashes, n_buckets, loss, time.time() - start_t))
    return loss


def run_testminsketch(y, y_scores, n_hashes, n_buckets, name):
    start_t = time.time()
    loss = test_oracle_count_min_sketch(y, y_scores, n_hashes, n_buckets, loss_function)
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
    argparser.add_argument("--n_examples", type=int, help="# of examples per .npy file", default=100000000)
    argparser.add_argument("--space_list", type=float, nargs='*', help="space in MB", default=[])
    argparser.add_argument("--n_hashes_list", type=int, nargs='*', help="number of hashes", required=True)
    argparser.add_argument("--n_workers", type=int, help="number of workers", default=10)
    argparser.add_argument("--word_data", action='store_true', default=False)
    argparser.add_argument("--aol_data", action='store_true', default=False)
    argparser.add_argument("--syn_data", action='store_true', default=False)
    argparser.add_argument("--test_as_valid", action='store_true', default=False)
    args = argparser.parse_args()

    name = 'dynamic_count_min'
 
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

    print('data loading time: %.1f sec' % (time.time() - start_t))

    if args.valid_results:
        if args.test_as_valid:
            key = 'test_output'
        else:
            key = 'valid_output'
        y_valid_ordered, y_valid_scores = order_y_wkey_list(y_valid, args.valid_results, key)

    # diff = 0
    # sum = 0
    # min = 10000000000
    # max = -100000000000
    # total = np.sum(y_valid_ordered)
    # with open(os.path.join(folder, 'oracle_results.csv'), "w+") as file:
    #     writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     for i in range(len(y_valid_ordered)):
    #         N = len(y_valid_ordered)
    #         true_freq = y_valid_ordered[i] / total
    #         est_freq = np.exp(y_valid_scores[i]) / total
    #         diff += true_freq - est_freq
    #         if i < 20:
    #             print("true_freq: " + str(true_freq) + " est_freq: " + str(est_freq) + " diff: " + str(true_freq - est_freq))
    #         sum += true_freq
    #         if np.abs(true_freq - est_freq) > max:
    #             max = np.abs(true_freq - est_freq)
    #         if np.abs(true_freq - est_freq) < min:
    #             min = np.abs(true_freq - est_freq)
    #         writer.writerow([str(true_freq), str(est_freq), str(true_freq - est_freq)])

    #     print("sum true frequency " + str(sum) + " avg diff = " + str(total * diff / len(y_valid_ordered)) + " min diff " + str(total*min) + " max diff " + str(total*max))

    # exit(0)

    if args.test_results:
        key = 'test_output'
        y_test_ordered, y_test_scores = order_y_wkey_list(y_test, args.test_results, key)

    nh_all = []
    nb_all = []
    nh_all_dyn = []
    nb_all_dyn = []
    for space in args.space_list:
        for n_hash in args.n_hashes_list:
            n_cmin_buckets = int(space * 1e6 / (n_hash * 4))
            n_dynamic_buckets = int(space * 1e6/ (n_hash * 4))
            nh_all.append(n_hash)
            nb_all.append(n_cmin_buckets)
            nh_all_dyn.append(n_hash)
            nb_all_dyn.append(n_dynamic_buckets)

            # (100, 500, 1000) : MB space 0.000084 0.0005 0.001

    # start_t = time.time()
    # pool = Pool(args.n_workers)
    # name = "dynamic count min"
    # results = pool.starmap(run_dynamicsketch, zip(repeat(y_valid_ordered), repeat(y_valid_scores), nh_all, nb_all, repeat(name)))
    # pool.close()
    # pool.join()
    
    # valid_results = results
    # rshape = (len(args.space_list),  len(args.n_hashes_list))
    # valid_results = np.reshape(valid_results, rshape)
    # nh_all = np.reshape(nh_all, rshape)
    # nb_all = np.reshape(nb_all, rshape)

    # log_str += '==== valid_results ====\n'
    # for i in range(len(valid_results)):
    #     log_str += 'space: %.2f\n' % args.space_list[i]
    #     for j in range(len(valid_results[i])):
    #         log_str += '%s: , # hashes %d, # buckets %d - \tloss %.2f\n' % \
    #             (name,  nh_all[i,j], nb_all[i,j], valid_results[i,j])
    # log_str += 'param search done -- time: %.2f sec\n' % (time.time() - start_t)

    # np.savez(os.path.join(folder, args.save+'_valid'),
    #     command=command,
    #     loss_all=valid_results,
    #     n_hashes=nh_all,
    #     n_buckets=nb_all,
    #     space_list=args.space_list,
    # )

    # log_str += '==== best parameters ====\n'
    # rshape = (len(args.space_list), -1)
    # best_param_idx = np.argmin(valid_results.reshape(rshape), axis=1)
    # best_n_buckets = nb_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    # best_n_hashes  = nh_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    # best_valid_loss  = valid_results.reshape(rshape)[np.arange(rshape[0]), best_param_idx]

    # print(best_n_buckets)
    # print(best_n_hashes)
    # parms = ""
    # for i in range(len(best_valid_loss)):
    #     parms += 'space: %.2f,  n_buckets %d, n_hashes %d - \tloss %.2f\n' % \
    #         (args.space_list[i], best_n_buckets[i], best_n_hashes[i], best_valid_loss[i])
    # print(parms)


    # name = 'dynamic count sketch'
    # # test data using best parameters
    # pool = Pool(args.n_workers)
    # test_results_dynamic = pool.starmap(
    #     run_dynamicsketch, zip(repeat(y_valid_ordered), repeat(y_valid_scores), 
    #     nh_all_dyn, nb_all_dyn, repeat(name)))
    # pool.close()
    # pool.join()

    # name = 'count sketch'
    # # test data using best parameters
    # pool = Pool(args.n_workers)
    # test_results = pool.starmap(
    #     run_countsketch, zip(repeat(y_valid_ordered), 
    #     nh_all, nb_all, repeat(name)))
    # pool.close()
    # pool.join()
    

    
    
    # Count sketch and dynamic count sketch 
    
    pool = Pool(args.n_workers)
    test_results_dynamic = pool.starmap(
        run_testsketch, zip(repeat(y_valid_ordered), repeat(y_valid_scores), 
        nh_all, nb_all, repeat('test_sketch')))
    pool.close()
    pool.join()

    pool = Pool(args.n_workers)
    test_results = pool.starmap(
        run_countsketch, zip(repeat(y_valid_ordered), 
        nh_all, nb_all, repeat('count_sketch')))
    pool.close()
    # pool.join()

    print("Percentage improvement: ")
    for i in range(len(test_results)):
        percentage = test_results[i] / test_results_dynamic[i]
        print(" " + str(np.floor((percentage - 1.0) * 100)))
    

    # with open(os.path.join(folder, args.save+'.log'), 'w') as f:
    #     f.write(log_str)

    # np.savez(os.path.join(folder, args.save+'_test'),
    #     command=command,
    #     loss_all=test_results,
    #     n_hashes=best_n_hashes,
    #     n_buckets=best_n_buckets,
    #     space_list=args.space_list,
    # )
  
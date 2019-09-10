import os
import sys
import time
import argparse

import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    print('cannot import matplotlib!')

from utils import get_data
from collections import defaultdict


def compute_avg_loss(counts, y, y_buckets):
    assert np.sum(counts) == np.sum(y), 'counts do not have all the flows!'
    assert len(y) == len(y_buckets)
    if len(y) == 0:
        return 0    # avoid division of 0
    loss = 0
    for i in range(len(y)):
        loss += np.abs(y[i] - counts[y_buckets[i]]) * y[i]
    return loss / len(y)

def plot_buckets(counts, show_option, title=''):
    ax = plt.figure().gca()
    if show_option == 'bar':
        ax.bar(range(len(counts)), counts)
    else:
        ax.plot(range(len(counts)), counts)
    ax.set_ylim([0, np.max(counts)+100])
    ax.set_xlabel('bucket id')
    ax.set_ylabel('# packets')
    ax.set_title(title)
    plt.show()

def greedy(y, n_buckets):
    counts = np.zeros(n_buckets)
    y_buckets = []
    for i in range(len(y)):
        idx = np.argmin(counts)
        counts[idx] += y[i]
        y_buckets.append(idx)

    loss = compute_avg_loss(counts, y, y_buckets)
    return counts, loss

def random_hash(y, n_buckets):
    counts = np.zeros(n_buckets)
    y_buckets = np.random.choice(np.arange(n_buckets), size=len(y))
    for i in range(len(y)):
        counts[y_buckets[i]] += y[i]
    loss = compute_avg_loss(counts, y, y_buckets)
    return counts, loss, y_buckets

def random_hash_avg(y, n_buckets, n_avg):
    counts_all = []
    loss_all = []
    for _ in range(n_avg):
        counts, loss, _ = random_hash(y, n_buckets)
        counts_all.append(counts)
        loss_all.append(loss)
    return counts_all, np.mean(loss_all)

def count_min(y, n_buckets, n_hash):
    if len(y) == 0:
        return 0    # avoid division of 0

    counts_all = np.zeros((n_hash, n_buckets))
    y_buckets_all = np.zeros((n_hash, len(y)), dtype=int)
    for i in range(n_hash):
        counts, loss, y_buckets = random_hash(y, n_buckets)
        counts_all[i] = counts
        y_buckets_all[i] = y_buckets

    loss = 0
    for i in range(len(y)):
        y_est = np.min([counts_all[k, y_buckets_all[k, i]] for k in range(n_hash)])
        loss += np.abs(y[i] - y_est) * y[i]
    return loss / len(y)


def random_hash_with_sign(y, n_buckets):
    '''
    assign items in y into n_buckets, randomly pick a sign for each item
    '''
    counts = np.zeros(n_buckets)
    y_buckets = np.random.choice(np.arange(n_buckets), size=len(y))
    y_signs = np.random.choice([-1, 1], size=len(y))
    for i in range(len(y)):
        counts[y_buckets[i]] += (y[i] * y_signs[i])
    return counts, y_buckets, y_signs

def count_sketch(y, n_buckets, n_hash):
    if len(y) == 0:
        return 0    # avoid division of 0

    counts_all = np.zeros((n_hash, n_buckets))
    y_buckets_all = np.zeros((n_hash, len(y)), dtype=int)
    y_signs_all = np.zeros((n_hash, len(y)), dtype=int)
    for i in range(n_hash):
        counts, y_buckets, y_signs = random_hash_with_sign(y, n_buckets)
        counts_all[i] = counts
        y_buckets_all[i] = y_buckets
        y_signs_all[i] = y_signs

    loss = 0
    for i in range(len(y)):
        y_est = np.median(
            [y_signs_all[k, i] * counts_all[k, y_buckets_all[k, i]] for k in range(n_hash)])
        loss += np.abs(y[i] - y_est) * y[i]
    return loss / len(y)

def cutoff_countsketch(y, n_buckets, b_cutoff, n_hashes):
    assert b_cutoff <= n_buckets, 'bucket cutoff cannot be greater than n_buckets'
    counts = np.zeros(n_buckets)
    if len(y) == 0:
        return 0            # avoid division of 0

    y_buckets = []
    for i in range(b_cutoff):
        if i >= len(y):
            break           # more unique buckets than # flows
        counts[i] += y[i]   # unique bucket for each flow
        y_buckets.append(i)

    loss_cf = compute_avg_loss(counts[:b_cutoff], y[:b_cutoff], y_buckets)
    loss_cs = count_sketch(y[b_cutoff:], n_buckets - b_cutoff, n_hashes)

    loss_avg = (loss_cf * b_cutoff + loss_cs * (len(y) - b_cutoff)) / len(y)
    print('\tloss_cf %.2f\tloss_rd %.2f\tloss_avg %.2f' % (loss_cf, loss_cs, loss_avg))

    space = b_cutoff * 4 * 2 + (n_buckets - b_cutoff) * n_hashes * 4
    return loss_avg, space

def cutoff_countsketch_wscore(y, scores, score_cutoff, n_cs_buckets, n_hashes):
    if len(y) == 0:
        return 0            # avoid division of 0

    y_ccm = y[scores >  score_cutoff]
    y_cs  = y[scores <= score_cutoff]

    loss_cf = 0  # put y_ccm into cutoff buckets, no loss
    loss_cm = count_sketch(y_cs, n_cs_buckets, n_hashes)

    assert len(y_ccm) + len(y_cs) == len(y)
    loss_avg = (loss_cf * len(y_ccm) + loss_cm * len(y_cs)) / len(y)
    print('\tloss_cf %.2f\tloss_rd %.2f\tloss_avg %.2f' % (loss_cf, loss_cm, loss_avg))

    space = len(y_ccm) * 4 * 2 + n_cs_buckets * n_hashes * 4
    return loss_avg, space


def round_robin(y, n_buckets):
    counts = np.zeros(n_buckets)
    y_buckets = []
    for i in range(len(y)):
        idx = i % n_buckets
        counts[idx] += y[i]
        y_buckets.append(idx)

    loss = compute_avg_loss(counts, y, y_buckets)
    return counts, loss

def cutoff_random(y, n_buckets, b_cutoff, n_avg):
    assert b_cutoff <= n_buckets, 'bucket cutoff cannot be greater than n_buckets'
    counts = np.zeros(n_buckets)
    if len(y) == 0:
        return counts, 0    # avoid division of 0

    y_buckets = []
    for i in range(b_cutoff):
        if i >= len(y):
            break           # more unique buckets than # flows
        counts[i] += y[i]   # unique bucket for each flow
        y_buckets.append(i)

    loss_cf = compute_avg_loss(counts[:b_cutoff], y[:b_cutoff], y_buckets)
    counts_all, loss_rd = random_hash_avg(y[b_cutoff:], n_buckets - b_cutoff, n_avg)
    counts[b_cutoff:] = counts_all[0]

    loss_avg = (loss_cf * b_cutoff + loss_rd * (len(y) - b_cutoff)) / len(y)
    print('\tloss_cf %.2f\tloss_rd %.2f\tloss_avg %.2f' % (loss_cf, loss_rd, loss_avg))
    return counts, loss_avg

def cutoff_countmin(y, n_buckets, b_cutoff, n_hashes):
    assert b_cutoff <= n_buckets, 'bucket cutoff cannot be greater than n_buckets'
    counts = np.zeros(n_buckets)
    if len(y) == 0:
        return 0            # avoid division of 0

    y_buckets = []
    for i in range(b_cutoff):
        if i >= len(y):
            break           # more unique buckets than # flows
        counts[i] += y[i]   # unique bucket for each flow
        y_buckets.append(i)

    loss_cf = compute_avg_loss(counts[:b_cutoff], y[:b_cutoff], y_buckets)
    loss_cm = count_min(y[b_cutoff:], n_buckets - b_cutoff, n_hashes)

    loss_avg = (loss_cf * b_cutoff + loss_cm * (len(y) - b_cutoff)) / len(y)
    print('\tloss_cf %.2f\tloss_rd %.2f\tloss_avg %.2f' % (loss_cf, loss_cm, loss_avg))

    space = b_cutoff * 4 * 2 + (n_buckets - b_cutoff) * n_hashes * 4
    return loss_avg, space

def cutoff_countmin_wscore(y, scores, score_cutoff, n_cm_buckets, n_hashes):
    if len(y) == 0:
        return 0            # avoid division of 0

    y_ccm = y[scores >  score_cutoff]
    y_cm  = y[scores <= score_cutoff]

    loss_cf = 0  # put y_ccm into cutoff buckets, no loss
    loss_cm = count_min(y_cm, n_cm_buckets, n_hashes)

    assert len(y_ccm) + len(y_cm) == len(y)
    loss_avg = (loss_cf * len(y_ccm) + loss_cm * len(y_cm)) / len(y)
    print('\tloss_cf %.2f\tloss_rd %.2f\tloss_avg %.2f' % (loss_cf, loss_cm, loss_avg))

    space = len(y_ccm) * 4 * 2 + n_cm_buckets * n_hashes * 4
    return loss_avg, space

def cutoff_lookup(x, y, n_cm_buckets, n_hashes, d_lookup, y_cutoff, sketch='CountMin'):
    if len(y) == 0:
        return 0            # avoid division of 0

    y_ccm = []
    y_cm = []
    for i in range(len(y)):
        if x[i] in d_lookup:
            if d_lookup[x[i]] > y_cutoff:
                y_ccm.append(y[i])
            else:
                y_cm.append(y[i])
        else:
            y_cm.append(y[i])

    loss_cf = 0 # put y_ccm into cutoff buckets, no loss
    if sketch == 'CountMin':
        loss_cm = count_min(y_cm, n_cm_buckets, n_hashes)
    elif sketch == 'CountSketch':
        loss_cm = count_sketch(y_cm, n_cm_buckets, n_hashes)
    else:
        assert False, "unknown sketch type"

    assert len(y_ccm) + len(y_cm) == len(y)
    loss_avg = (loss_cf * len(y_ccm) + loss_cm * len(y_cm)) / len(y)
    print('\tloss_cf %.2f\tloss_rd %.2f\tloss_avg %.2f' % (loss_cf, loss_cm, loss_avg))
    print('\t# uniq', len(y_ccm), '# cm', len(y_cm))

    space = len(y_ccm) * 4 * 2 + n_cm_buckets * n_hashes * 4
    return loss_avg, space

def cutoff_random_wscore(y, scores, n_buckets, uniq_bucket_scores, n_avg):
    '''
    uniq_bucket_scores: the right boundaries of each bucket
    '''
    assert np.array_equal(uniq_bucket_scores, np.sort(uniq_bucket_scores)[::-1]) # descending order
    n_uniq_buckets = len(uniq_bucket_scores)
    idx = scores >= uniq_bucket_scores[-1]
    scores_uniq_b = scores[idx]
    y_uniq_b = y[idx]
    y_rand = y[~idx]

    y_buckets = unique_bucket_mapping(scores_uniq_b, uniq_bucket_scores)
    assert np.max(y_buckets) < n_uniq_buckets

    counts = np.zeros(n_buckets)
    for i in range(len(y_buckets)):
        counts[y_buckets[i]] += y_uniq_b[i]   # unique bucket for each flow

    from collections import Counter
    #print('\tcollisions:')
    b_c = []
    for b, count in Counter(y_buckets).items():
        if count > 1:
            #print('\t', b, count)
            b_c.append((b, count))

    #np.savez('debug_bc.npz',
    #    scores_uniq_b=scores_uniq_b,
    #    bucket_scores=uniq_bucket_scores,
    #    y_buckets=y_buckets,
    #    )

    print('\t# colliding buckets', len(b_c))
    print('\t# empty buckets', np.sum(counts[:n_uniq_buckets] == 0))

    loss_cf = compute_avg_loss(counts[:n_uniq_buckets], y[idx], y_buckets)
    counts_all, loss_rd = random_hash_avg(y[~idx], n_buckets - n_uniq_buckets, n_avg)
    loss_avg = (loss_cf * np.sum(idx) + loss_rd * np.sum(~idx)) / len(y)
    print('\tloss_cf %.2f\tloss_rd %.2f\tloss_avg %.2f' % (loss_cf, loss_rd, loss_avg))

    counts[n_uniq_buckets:] = counts_all[0]
    return counts, loss_avg

def unique_bucket_mapping(flow_scores, bucket_scores):
    ''' create mapping from score to bucket id'''
    return np.digitize(flow_scores, bucket_scores)

def order_y_wkey(y, results, key, n_examples=0):
    print('loading results from %s' % results)
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


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--data", type=str, nargs='*', help="list of input .npy data", required=True)
    argparser.add_argument("--save", type=str, help="post to save the results", default='')
    argparser.add_argument("--seed", type=int, help="random state for sklearn", default=69)
    argparser.add_argument("--n_examples", type=int, help="# of examples per .npy file", default=10000000)
    argparser.add_argument("--n_buckets", type=int, nargs='*', help="number of buckets", required=True)
    argparser.add_argument("--n_avg", type=int, help="number of runs for random hash", default=20)
    argparser.add_argument("--simulation", action='store_true', help="make all # packets 1", default=False)
    argparser.add_argument("--show_buckets", type=str, help="visualize bucket filling", default='')
    argparser.add_argument("--random", action='store_true', help="random hash", default=False)
    args = argparser.parse_args()

    assert not (len(args.n_buckets) > 1 and args.save == ''), "use --save"

    np.random.seed(args.seed)
    x, y = get_data(args.data, feat_idx=np.arange(8), n_examples=args.n_examples)

    if args.simulation:
        y = np.ones(len(x))

    y = sorted(y)[::-1] # Note: order of x is different now

    if len(args.n_buckets) == 1:
        n_buckets = args.n_buckets[0]
        start_t = time.time()
        if args.random:
            name = 'random'
            counts, loss = random_hash_avg(y, n_buckets, args.n_avg)
        else:
            name = 'greedy'
            counts, loss = greedy(y, n_buckets)
        total_time = time.time() - start_t

        print('==== %s ====' % name)
        print('multi-set loss %.2f' % loss)
        print('bucket variance %.2f' % np.var(counts))
        print('Total time %.1f sec' % total_time)

        if args.show_buckets:
            plot_buckets(counts, args.show_buckets)
    else:
        start_t = time.time()
        results = defaultdict(list)
        for n_buckets in args.n_buckets:
            print('n buckets ', n_buckets)
            counts_rd, loss_rd = random_hash_avg(y, n_buckets, args.n_avg)
            counts_gd, loss_gd = greedy(y, n_buckets)
            results['counts_random'].append(counts_rd)
            results['counts_greedy'].append(counts_gd)
            results['loss_random'].append(loss_rd)
            results['loss_greedy'].append(loss_gd)
            print('\ttime %.1f sec' % (time.time() - start_t))

        np.savez('rgd_' + args.save + '.npz',
            command=' '.join(sys.argv),
            results=results)

        total_time = time.time() - start_t
        print('Total time %.1f sec' % total_time)


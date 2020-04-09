import os
import sys
import time
import argparse
import random

import numpy as np

def compute_avg_loss(counts, y, y_buckets):
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
    # TODO(sss): remove the loss calculation here; useless
    loss = compute_avg_loss(counts, y, y_buckets)
    return counts, loss, y_buckets

def random_hash_with_bucket_weights(y, weights, n_buckets):
    counts = np.zeros(n_buckets)
    sum_weights = np.zeros(n_buckets)
    y_buckets = np.random.choice(np.arange(n_buckets), size=len(y))
    for i in range(len(y)):
        counts[y_buckets[i]] += y[i]
        sum_weights[y_buckets[i]] += weights[i]
    return counts, sum_weights, y_buckets


def random_hash_with_scores(y, y_scores, n_buckets):
    counts = np.zeros(n_buckets)
    scores = np.zeros(n_buckets)
    y_buckets = np.random.choice(np.arange(n_buckets), size=len(y))
    for i in range(len(y)):
        counts[y_buckets[i]] += y[i]
        scores[y_buckets[i]] += y_scores[i]
    return counts, scores, y_buckets

def random_hash_avg(y, n_buckets, n_avg):
    counts_all = []
    loss_all = []
    for _ in range(n_avg):
        counts, loss, _ = random_hash(y, n_buckets)
        counts_all.append(counts)
        loss_all.append(loss)
    return counts_all, np.mean(loss_all)

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


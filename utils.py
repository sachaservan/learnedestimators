# TODO: credit this to ChenYu's implementation 

import os
import numpy as np
import subprocess
import pickle
from collections import deque

# begin workaround for bug in latest numpy version
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# end workaround

def save_pickle(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def get_stat(data_name, data_x, data_y):
    s = 'statistics for %s\n' % data_name
    s += 'data #: %d, shape %s\n' % (len(data_x), str(np.asarray(data_x).shape))
    if len(data_y) > 0:
        s += 'positive ratio: %.5f, max %f, min %f\n' % \
            (np.mean(data_y), np.max(data_y), np.min(data_y))
    s += '\n'
    print(s)
    return s

def feat_to_string(v):
    return ''.join([str(int(i)).zfill(3) for i in v])

def string_to_feat(s):
    assert len(s) % 3 == 0
    feat = []
    for i in range(0, len(s), 3):
        feat.append(float(s[i:i+3]))
    return np.asarray(feat)

def git_log():
    return subprocess.check_output(['git', 'log', '-n', '1']).decode('utf-8')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ExpoAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, weight):
        self.reset()
        self.weight = weight

    def reset(self):
        self.val = 0
        self.avg = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.count += 1
        self.avg = self.avg * self.weight + val * (1 - self.weight)


def format_data_wports(data_x, n_examples):
    data_ip = decimal2binary(data_x[:n_examples, np.arange(8)])
    data_proto = data_x[:n_examples, 8].reshape(-1, 1)
    data_srcport = uint16_to_binary(data_x[:n_examples, 9].reshape(-1, 1))
    data_dstport = uint16_to_binary(data_x[:n_examples, 10].reshape(-1, 1))
    return np.concatenate((data_ip, data_srcport, data_dstport, data_proto), axis=1)

def get_data_args(data_list, feat_idx, args):
    return get_data(data_list, feat_idx, args.n_examples, args.bin_feat, args.hh_thr, args.port_hiddens)

def get_data(data_list, feat_idx, n_examples, binary=False, hh_thr=0, port_hiddens=[]):
    if not binary:
        data_x = np.array([]).reshape(0, len(feat_idx))
    else:
        if port_hiddens:
            data_x = np.array([]).reshape(0, 8*8+2*16+1)   # src, dst, ip
        else:
            data_x = np.array([]).reshape(0, len(feat_idx)*8)
    data_y = np.array([])
    for data in data_list:
        data = np.load(data).item()
        if not binary:
            data_x = np.concatenate((data_x, data['x'][:n_examples, feat_idx] / 256.0))
        else:
            if port_hiddens:
                # NOTE: order of the features are changed!
                data_b = format_data_wports(data['x'], n_examples)
            else:
                data_b = decimal2binary(data['x'][:n_examples, feat_idx])
            data_x = np.concatenate((data_x, data_b))

        if hh_thr > 0:
            data_y = np.concatenate((data_y, data['y'][:n_examples] > hh_thr))
        else:
            data_y = np.concatenate((data_y, data['y'][:n_examples]))
    return data_x, data_y

def data_to_string(data):
    ip = feat_to_string(data[:8])
    proto = str(int(data[8])).zfill(3)
    ports = ''.join([str(int(i)).zfill(5) for i in data[9:]])
    return ip + proto + ports

def get_data_str_with_ports(data):
    data = np.load(data).item()
    data_x = data['x']
    data_y = data['y']

    data_x_str = []
    for xi in data_x:
        data_x_str.append(data_to_string(xi))
    return data_x_str, data_y

def get_data_str_with_ports_list(data_list):
    data_x = []
    data_y = np.array([])
    for dpath in data_list:
        x, y = get_data_str_with_ports(dpath)
        data_x += x
        data_y = np.concatenate((data_y, y))
    return data_x, data_y

def get_data_list_args(data_list, feat_idx, args):
    return get_data_list(data_list, feat_idx, args.n_examples, args.bin_feat, args.hh_thr, args.port_hiddens)

def get_data_list(data_list, feat_idx, n_examples, binary=False, hh_thr=0, port_hiddens=[]):
    data_x = []
    data_y = []
    for data in data_list:
        data = np.load(data).item()
        if not binary:
            data_x.append(data['x'][:n_examples, feat_idx] / 256.0)
        else:
            if port_hiddens:
                # NOTE: order of the features are changed!
                data_b = format_data_wports(data['x'], n_examples)
            else:
                data_b = decimal2binary(data['x'][:n_examples, feat_idx])
            data_x.append(data_b)

        if hh_thr > 0:
            data_y.append(data['y'][:n_examples] > hh_thr)
        else:
            data_y.append(data['y'][:n_examples])
    return data_x, data_y

def get_data_series(data_path, series_path, n_examples=0):
    data = np.load(data_path).item()
    if n_examples == 0:
        n_examples = len(data['y'])

    data_x = format_data_wports(data['x'], n_examples)
    data_y = data['y'][:n_examples]

    data = np.load(series_path)
    data_s = data['series'][:n_examples]
    data_s_counts = np.sum(data_s, axis=1)

    check_idx = np.random.choice(n_examples, 10)
    assert np.array_equal(data_s_counts[check_idx], data_y[check_idx]), "something is wrong"
    data_s /= data_s_counts[:, np.newaxis]    # normalize time series

    return data_x, data_y, data_s

def get_data_series_all(data_paths, series_paths):
    data_x = np.array([]).reshape(0, 8*8+2*16+1)   # src, dst, ip
    data_y = np.array([])
    data_s = np.array([])
    for dp, sp in zip(data_paths, series_paths):
        x, y, s = get_data_series(dp, sp)
    # TODO finish this


def decimal2binary(x):
    return np.unpackbits(x.astype(np.uint8), axis=1)

def uint16_to_binary(x):
    assert len(x.shape) == 2 and x.shape[1] == 1
    return np.roll(np.unpackbits(x.astype(np.uint16).view(np.uint8), axis=1), 8, axis=1)

def binary2decimal(x):
    return np.packbits(x, axis=1)

def keep_latest_files(path, n_keep):
    def sorted_ls(path):
        mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
        return list(sorted(os.listdir(path), key=mtime))

    files = sorted_ls(path)
    if len(files) < n_keep:
        return

    del_list = files[0:(len(files)-n_keep)]
    for dfile in del_list:
        os.remove(path + dfile)

# for curriculum learning

def get_thr_pctg_map(y, thr_list):
    return np.array([len(y[y > thr]) / len(y) for thr in thr_list])

class Scheduler(object):
    """Return a new hh threshold if training loss saturates"""
    def __init__(self, data_y, init_pctg, weight, n_ep, ratio):
        self.thr_list  = np.linspace(0, 1000, 501)[::-1]
        self.pctg_map  = get_thr_pctg_map(data_y, self.thr_list)

        # get the first threshold exceeding curr_pctg
        idx = self.get_idx(init_pctg)
        self.thr = self.thr_list[idx]
        self.curr_pctg = self.pctg_map[idx]
        self.expAvg  = ExpoAverageMeter(weight)
        self.loss_q  = deque(maxlen=n_ep)
        self.p_ratio = ratio

    def get_idx(self, pctg_thr):
        idx = np.argmax(self.pctg_map >= pctg_thr)
        if self.pctg_map[idx] < pctg_thr:
            idx = -1
        return idx

    def reset(self):
        self.expAvg.reset()
        self.loss_q.clear()

    def update(self, loss):
        self.expAvg.update(loss)
        self.loss_q.append(loss)

        if len(self.loss_q) != self.loss_q.maxlen:
            return False
        else:
            self.progress = np.abs(self.expAvg.avg - self.loss_q[0]) / self.loss_q[0]
            #print('\t p:', self.progress, 'ratio:', self.p_ratio)
            if self.progress > self.p_ratio:
                return False # good progress
            else:
                # get the next threshold
                idx = self.get_idx(self.curr_pctg+0.01)
                self.thr = self.thr_list[idx]
                self.curr_pctg = self.pctg_map[idx]
                self.reset()
                return True


def get_data_aol(data_list):
    c = Counter()
    for data in data_list:
        counter = load_pickle(data)
        print('%s ... # items %d' % (data, len(counter)))
        c.update(counter)

    x, y = zip(*c.most_common())
    return np.asarray(x), np.asarray(y)

def get_data_aol_by_day(data_path):
    data = np.load(data_path)
    return data['queries'], data['counts']

def get_data_aol_by_days(data_list):
    c = Counter()
    for data in data_list:
        print('loading %s' % data)
        x, y = get_data_aol_by_day(data)
        c.update(dict(zip(x, y)))

    x, y = zip(*c.most_common())
    return np.asarray(x), np.asarray(y)

def get_data_aol_feat(data_path):
    data = np.load(data_path)
    query_char_ids = data['query_char_ids']
    counts  = data['counts']
    q_lens  = data['query_lens']

    print('queries', query_char_ids.shape)
    print('counts', counts.shape)
    print('q_lens', q_lens.shape)

    x = np.concatenate((q_lens.reshape(-1, 1), query_char_ids), axis=1)
    y = counts
    assert len(x) == len(y)
    return x, y

def get_data_aol_feat_list(data_paths):
    x = np.array([]).reshape(0, 61)
    y = np.array([])
    for dpath in data_paths:
        xi, yi = get_data_aol_feat(dpath)
        x = np.concatenate((x, xi))
        y = np.concatenate((y, yi))
    return x, y

def get_data_aol_query(data_path):
    data = np.load(data_path)
    queries = data['queries']
    counts  = data['counts']

    print('queries', queries.shape)
    print('counts', counts.shape)

    assert len(queries) == len(counts)
    return queries, counts

def get_data_aol_query_list(data_paths):
    queries = np.array([])
    counts  = np.array([])
    for dpath in data_paths:
        qi, ci = get_data_aol_query(dpath)
        queries = np.concatenate((queries, qi))
        counts = np.concatenate((counts, ci))
        print(dpath)
    return queries, counts

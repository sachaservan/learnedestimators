import os
import sys
import time
import argparse
import random
import math
import numpy as np

def log(x, b):
    '''
    convinience function for computing log in arbitrary base b
    '''
    return  math.log(x) / math.log(b)

def random_hash(y, n_buckets):
    '''
    randomly assign items in y into n_buckets
    '''
    counts = np.zeros(n_buckets)
    y_buckets = np.random.choice(np.arange(n_buckets), size=len(y))
    for i in range(len(y)):
        counts[y_buckets[i]] += y[i]
    return counts, y_buckets

def random_hash_with_sign(y, n_buckets):
    '''
    randomly assign items in y into n_buckets, randomly pick a sign for each item
    '''
    counts = np.zeros(n_buckets)
    y_buckets = np.random.choice(np.arange(n_buckets), size=len(y))
    y_signs = np.random.choice([-1, 1], size=len(y))
    for i in range(len(y)):
        counts[y_buckets[i]] += (y[i] * y_signs[i])
    return counts, y_buckets, y_signs
    

def second_moment_estimate(y):
    ''' 
    computes an estimate of the raw second momement in a stream of data
    '''
    iters = 10 # simultates 10 sign hash functions 
    est = 0
    for i in range(iters):
        y_signs = np.random.choice([-1, 1], size=len(y))
        est += (np.inner(y_signs, y))**2 # inner product squared

    return est / iters


def compute_sum_estimate(values, eps, n_reg=16):
    ''' 
    computes a sum of values in a stream; without double counting 
    error is bounded by by eps if n_reg is chosen accordingly 
    '''
    estimated_sum = 0
    logbase = 1 + eps
    exp = np.array([int(round(log(x, logbase))) for x in values]) # compute rounding of log(x) with base logbase
    exp.sort() # sort the array lowest to highest 

    i = 0
    while i < len(exp):
        n = (exp == exp[i]).sum() # count occurences of this element
        hll_count = hyperloglogsimulate(n, n_reg) # compute hyperloglog estimate (simulated for n elements)
        estimated_sum += (logbase ** exp[i]) * hll_count # add to the running sum

        # move on to next highest value
        k = i
        while k < len(exp) and exp[k] == exp[i]:
            k += 1
        i = k


    # total space in bytes: number of "groupings" times the n
    # umber of regiseters in HLL (assume 1 byte per register)
    u = len(np.unique(exp))
    space = u * n_reg  # space in bytes

    return estimated_sum, space

def hyperloglogsimulate(n, m): 
    '''
    computes the estimated number of unique values for a set of n values using m 
    byte registers. see https://en.wikipedia.org/wiki/HyperLogLog
    '''

    assert(math.log(m, 2).is_integer()) # m must be a power of two

    lm = int(math.log(m, 2))

    assert(lm < 64) # the "hash" assumes 128 bits of which the first lm are used

    # compute optimal constant based on number of registers
    if m == 16:
        a = 0.673
    elif m == 32:
        a = 0.697
    elif m == 64:
        a = 0.709
    else:
        a = 0.7213 / (1 + 1.079/m) # correction value

    registers = np.zeros(m, dtype=int)

    # don't take about the item b/c y is assumed to be sorted
    # and only contain unique values
    for _ in range(n): 
        # y contains only unique values; just pick a random hash output for it
        bval = np.random.choice(np.arange(2), size=m) 
        regidx = 0
        for i in range(0, lm): # compute register index
            regidx += (1 << i) * bval[i]

        msbidx = 0 # index (+1) of most significant bit in the hash
        for i in range(len(bval[lm:])):
            if bval[lm:][i] == 1:
                msbidx = i + 1
                break

        # set to the max beteween current register and new value
        if msbidx > registers[regidx]:
            registers[regidx] = msbidx
    
    # hyperloglog estimate
    z = 0.0
    for i in range(len(registers)): 
        s = (1 << registers[i])
        z += 1.0/s

    E = a * (m**2) / z 

    if E <= (5/2) * m: 
        v = 0
        for i in range(m):
            if registers[i] == 0:
                v += 1
        if v != 0:
            E = m * math.log(m/v, 2) # small range correction 

        # TODO: implement large range correction 

    return E 
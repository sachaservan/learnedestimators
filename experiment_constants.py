#!/usr/bin/env python
# encoding: utf-8
"""
Constants.py
"""

ALGO_TYPE_CUTOFF_AND_ORACLE = "LEARNED_CUTOFF_AND_ORACLE"

CUTOFF_FRAC_TO_TEST = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# standard deviation in the oracle error (error sampled from normal distibution)
# when evaluating the synthetic data 
SYNTHETIC_DATA_ORACLE_SPACE_FACTOR_TO_TEST = [0.004, 0.04, 0.4]

MODEL_SIZE_AOL = 0.0152 # amortized model size 
MODEL_SIZE_IP =  0.0031

COUNT_SKETCH_OPTIMAL_N_HASH = 3
COUNT_MIN_OPTIMAL_N_HASH = 2
N_REGISTERS_FOR_HLL = 64
DISTINCT_ELEMENT_SUM_ESTIMATE_ERROR = 0.1
CUTOFF_SPACE_COST_FACTOR = 2

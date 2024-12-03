# mwu.py
"""
Module containing all Mann-Whitney U test related functions used for out-of-distribution testing.
"""

from scipy.stats import mannwhitneyu
import pandas as pd
import math 
import torch
import numpy as np

def mwu_certainty_dist_test_internal(test_sample,in_dist_sample,nan_policy='omit'):
    """ Function to be run using a test_sample against the known in_distribution sample
    """
    try:
        return mannwhitneyu(x=test_sample,y=in_dist_sample,nan_policy=nan_policy)
    except ValueError:
        return (np.nan,np.nan)

def get_global_mwu(tpcs,fpcs,n=30,tpr_threshold=.95):
    """ Function to get the MWU test results for the global TP and FP distributions

    Parameters:
    ------------------------
    :tpcs: np.ndarray of True Postitive certainty scores
    :fpcs: np.ndarray of FP certainty scores
    """
    accept=0
    N = len(fpcs)
    replace = (N<n)
    S = math.floor(1.5*(N//n)) if (2*n<=N) else 15
    prob=torch.ones(fpcs.shape[0])/fpcs.shape[0]
    pvals=[]
    for i in range(S):
        idx =prob.multinomial(num_samples=n,replacement=replace)
        p= mannwhitneyu(x=tpcs.clone().flatten().detach().numpy(),y=fpcs[idx].clone().flatten().detach().numpy())
        pvals.append(p[1])
        try:
            t= p[1]>1-tpr_threshold
            accept+=t.sum()
        except ValueError:
            print('Encountered Value error and ambiguous p= {}'.format(p))
            for i in range(len(p)):
                if p[1][i]>1-tpr_threshold:
                    accept+=1
    pvals=torch.Tensor(np.array(pvals))
    #print('The final pvals is {}'.format(pvals))
    return dict({'MWU p-values':pvals,'OOD%':accept})

def mwu_certainty_dist_test_tranches(test_sample,in_dist_sample,tranche_size=30,tpr_threshold=0.95):
    """ Given a large test_sample, we run the mwu_certainty_dist_test on tranches of a proportional size to the test sample, and return an array of p-values from the MWU test"""
    if tranche_size<1:
        tranche_size=1
    N = test_sample.shape[0]
    I = math.floor(N/tranche_size)
    if I <1:
        I=1
    if (N>1) and (N<8):
        I=N
    if (N>8) and (I<8):
        I=8
    if len(test_sample.shape)>2:
        # In this circumstance, we suppose we have Bayesian samples occupying our first tranche
        # To compare, we will try to sample the fpcs
        return get_global_mwu(tpcs=in_dist_sample,fpcs=test_sample[-1],n=tranche_size,tpr_threshold=tpr_threshold)
    else:
        return get_global_mwu(tpcs=in_dist_sample,fpcs=test_sample,n=tranche_size,tpr_threshold=tpr_threshold)

# oodd.py
import torch
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import tqdm
import torch.nn.functional as F
import math
import gc
from math import asin,pi,tan,sin,cos,comb,floor
from scipy import stats
from .omicrons import omicron_fn
from tqdm import tqdm
import pickle
import os
from typing import Union,Optional
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_precision_recall_curve,
    binary_roc,
)
from torchmetrics.utilities.compute import auc

from pytorch_ood.detector import *
from pytorch_ood.utils import OODMetrics

def send_to_bot(x):
    """ Returns negative infinity, to be called internally in order to pickle ECDF functions
    Parameters
    ------
    x : any Type
    
    Returns
    -------
    -np.Inf
    """
    return -np.Inf

def int_check(x:float,gamma:float,delta:float)->int:
    """ Internal function meant to check if a value falls within an interval
    Parameters
    ----------
    x : float
    gamma : float
    delta : float

    Returns
    -------
    Int value of 1 or 0 for Boolean computation.
    """
    return 1 if (gamma< x and x <delta) else 0

def make_mat(dfs:pd.DataFrame,sample_size:int):
    """ Returns a torch.Tensor matrix to convert a dataframe with Bayesian sampled samples so that the rows correspond to the Bayesian samples of a sample
    
    Parameters
    ----------
    dfs: pandas.DataFrame
        Must be a data frame with a certainty_score column whose type is float, such that the sample_size divides the length of the 1-d array extracted from the pandas.DataFrame
    sample_size: int 
        Must factor through dfs.shape[0]
        
    Returns
    -------
    mat: torch.Tensor
        (1,1) tensor with shape (-1,sample_size) of certainty scores where the columns correspond to the ith Bayesian sample. 
    """
    long_string = np.array(dfs.certainty_score)
    mat = torch.Tensor(long_string).reshape(sample_size,-1).t()
    return mat

def certainty_score_params(certainty_score_sample:Union[torch.Tensor,np.ndarray]):
    """Function to return mean and standard deviation of a certainty score sample distribution"""
    if isinstance(certainty_score_sample,torch.Tensor):
        std,mean= torch.std_mean(certainty_score_sample,unbiased=False)
        return mean,std
    else:
        return np.mean(certainty_score_sample),np.std(certainty_score_sample)


def oodd_test_and_metrics_external(detector,detector_name,combined_dataloader:Optional[torch.utils.data.DataLoader]=None,X:Optional[torch.Tensor]=None,Y:Optional[torch.Tensor]=None,device='cpu',tpr_threshold=.95,delete_detector=True):
    """Function for applying an out of distribution detector to a dataloader or pair of X,Y data

    Parameters
    ------------
    :detector: OOD Detector from pytorch_ood
    :detector_name: string naming the detector
    :combined_dataloader: optional, contains input data and labels
    :X: optional torch.Tensor consisting of input features
    :Y: optional torch.Tensor consisting of labels for the data
    :device: torch.device indicating what cuda device to use, defaults to 'cpu'
    :tpr_threshold: cut-off threshold for analysis of TPR 
    :delete_detector: bool, default=True, deletes the detector from memory after use if True
    """
    try:
        oodd_experiment_and_metrics=dict()
        print("Working on device {}".format(device))
        if "Mahalanobis" in detector_name:
            detector.fit(combined_dataloader,device=device)
        # Since the detector is already provided, we just need to evaluate the detector
        metrics = OODMetrics()
        if combined_dataloader is not None:
            for x,y in combined_dataloader:
                metrics.update(detector(x.to(device)),y.to(device))
        else:
            metric.update(detector(X.to(device),y.to(device)))
        oodd_experiment_and_metrics.update(metrics.compute())
        fprname="FPR"+str(tpr_threshold).split('.')[-1]+"TPR"
        print("\n\nThe metrics for {} are :\n{}".format(detector_name,oodd_experiment_and_metrics))
        if delete_detector:
            del detector
            torch.cuda.empty_cache()
            gc.collect()
        return oodd_experiment_and_metrics
    except Exception as E:
        print("While attempting to get the OOD results for {}, the following error occured {}".format(detector_name,E))

def fpr_at_tpr(pred, target, tpr_threshold=0.95):
    """                                                                                                
    Calculate the False Positive Rate at a certain True Positive Rate                                  
                                                                                                       
    :pred: outlier scores                                                                        
    :target: target label
    :tpr_threshold: cut-off value
    """
    # results will be sorted in reverse order                                                          
    fpr, tpr, _ = binary_roc(pred, target)
    idx = torch.searchsorted(tpr, tpr_threshold)
    if idx == fpr.shape[0]:
        return fpr[idx - 1]
    return fpr[idx]

def oodd_test_and_metrics_internal(in_scores:torch.Tensor,out_scores:torch.Tensor,OODD_results:dict,tpr_threshold:float):
    """Gather the oodd_experiment and novel_oodd_metrics information for one of (currently) three internal tests and outputs a dictionary containing
    performance information. In particcular, the in_scores, out_scores, and OODD results are all computed prior to application of this function which returns the performance results of the test.

    Inputs:
    ----------------------
    :in_scores: torch.Tensor returning test scores for in-distribution data
    :out_scores: torch.Tensor returning test scores for data drawn from unknown distribution
    :OODD_results: torch.Tensor returning the detection test results for the data drawn from unknown distribution
    :tpr_threshold: float between 0 and 1 that calibrates the remaining tests and metrics; must be applied to the oodd test prior to using this function.
    
    Output:
    :oodd_experiment_and_metrics: dictionary containing the scores of the test, the OODD test result, detection error, AUROC, AUPR-IN, AUPR-OUT, and FPR@TPR_threshold
    """
    oodd_experiment_and_metrics=dict({"in_scores":in_scores.cpu(),
                             "out_scores":out_scores.cpu(),
                             "OODD":OODD_results,
                             "Detection_Error":1-(tpr_threshold+OODD_results)/2.0,})
   
    scores,labels = torch.cat([in_scores,out_scores]),torch.cat([torch.zeros(in_scores.shape[0]),torch.ones(out_scores.shape[0])])
    internal_results = oodd_metrics_internal(scores=scores,labels=labels,tpr_threshold=tpr_threshold)
    oodd_experiment_and_metrics.update(internal_results)
    print("Results are {}".format(internal_results))
    print("OODD: {}".format(oodd_experiment_and_metrics["OODD"]))
    return oodd_experiment_and_metrics


def oodd_metrics_internal(scores:torch.Tensor,labels:torch.Tensor,tpr_threshold:float,device='cpu'):
    """ We want to gather AUROC, AUPR IN, AUPR OUT, and FPR@95TPR from our detector, given inputs the oodd test scores and labels (known in vs unknown)
    Note: outputs y will need to be from a combined dataset with our in and out of distribution data. In the original pytorch_ood implementation all OODD has been given a negative label.  
    Parameters
    -----------------------------
    :scores: torch.Tensor consisting of the detector's scores
    :labels: torch.Tensor, in and out of distribution labels, defaults for 0 if in and 1 if out (the detector is classifying OUT of distribution examples)
    :tpr_threshold: float, cutoff value
    :device: indicates the cuda device to use for processing if one is available, defaults as cpu
    """
    scores=scores.to(device)
    labels=labels.to(device)
    print("The labels are {}".format(labels))
    scores, scores_idx = torch.sort(scores, stable=True)
    labels = labels[scores_idx]
    labels= labels.long()
    auroc = binary_auroc(scores, labels)

        # num_classes=None for binary                                                                  
    p, r, t = binary_precision_recall_curve(scores,labels)
    aupr_in = auc(r, p)

    p, r, t = binary_precision_recall_curve(-scores,1- labels)
    aupr_out = auc(r, p)
    if tpr_threshold>=1.0:
        tpr_threshold=1
    fprname = "FPR"+str(tpr_threshold).split('.')[-1]+"TPR"
    fpr = fpr_at_tpr(scores, labels,tpr_threshold)
    return {
        "AUROC": auroc.cpu(),
        "AUPR-IN": aupr_in.cpu(),
        "AUPR-OUT": aupr_out.cpu(),
        fprname: fpr.cpu(),
        }

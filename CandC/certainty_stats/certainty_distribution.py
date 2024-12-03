# certainty_distribution
import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
import torch

from tqdm import tqdm

from ..candc.competence import *

def find_certainty_dist_dataframe(test_data=False,**kwargs):
    """ Returns the certainty distributions (cert_dist) as a pandas data frame
    
    Parameters
    ----------
    :classification_cat: torch.Tensor| numpy.array | list 
        An array of integers or strings describing each indexed values underlying label
    :cat_predict: torch.Tensor | numpy.array | list
        Integers describing each indexed values
    :certainty_score: torch.Tensor | numpy.array
        The pointwise minimum certainty scores for corresponding sample information, ideally generated as the cert_scores from get_certainty()
    :test_data: bool
        Corresponds to whether the data being processed is test_data, or data drawn from the training/validation data set. Determines if predictive status should be set to "test" or left as is,
        defaults to False
    :is_bayesian: used
    kwargs: dict
        The only additional keyword argument that will be processed is 'predictions' if it is found; this will append the predictions
        to the output dataframe.
    Returns
    -------
    cert_dist : dictionary whose primary key, 'cert_dist' is a pd.Dataframe
        Dictionary whose primary argument is a pandas data frame where rows correspond to (flattened) samples, and columns describe the true label (classification_cat),
        the prediction of our model (predicted_label), the assigned certainty score (certainty_score), and the predictive status (predictive_status) of the corresponding sample, 
        i.e. whether it was a TP, a FP, or status unknown and sample is a 'test' sample.
    """
    if isinstance(kwargs['classification'],torch.Tensor):
        classification_cat = kwargs['classification'].detach().clone().long().numpy()
    else:
        classification_cat = kwargs['classification']
    if kwargs['is_bayesian']:
        repeat = kwargs['cat_predict'].shape[0]
        classification_cat = np.array(classification_cat)
        classification_cat = np.tile(kwargs['classification'],repeat)
        classification_cat = torch.Tensor(classification_cat).long().numpy()
    if isinstance(kwargs['cat_predict'],torch.Tensor):
        predicted_label = kwargs['cat_predict'].long().flatten().numpy()
    else:
        predicted_label = kwargs['cat_predict']
    if type(kwargs['cert_score'])==torch.Tensor:
        certainty_score = kwargs['cert_score'].detach().flatten().numpy()
    else:
        certainty_score = kwargs['cert_score'].reshape(-1)
    print("The length of classification_cat is {}\n the length of predictions is {}\n the length of certainty scores is {}".format(len(classification_cat),len(predicted_label),len(certainty_score)))
    cert_dist = pd.DataFrame({'classification_cat': classification_cat, 'prediction': predicted_label,'certainty_score': certainty_score},index=[n for n in range(len(classification_cat))])
    if test_data:
        cert_dist['predictive_status']=['test' for n in range(cert_dist.shape[0])]
    else:
        cert_dist['predictive_status']=['TP' if cert_dist.classification_cat.iloc[x]==cert_dist.prediction.iloc[x] else 'FP' for x in range(len(cert_dist))]
    return cert_dist

def dist_stats(contains_test=True,**kwargs):
    """ Returns a dictionary of multi-indexed data-frames describing statistics of certainty scores within predicted labels and across a model dataset,
    including the Mann-Whitney U test to be performed pairwise on the distributions of certainty scores by predictive status.
    
    Parameters
    ----------
    
    kwargs['cert_dist']: pandas.Dataframe
        Dataframe must contain certainty score distribution information, eg the certainty score distribution columns : {'classification_cat', 'prediction', 'certainty_score', 'predictive_status'}
        Dataframe may optionally contain a column for 'predictions', which should be stored as an np.array, in order to gather additional statistics for out of certainty distribution
    observed_label: numpy.array
        A corresponding array that is used to select each samples observed label
    contains_test : bool
        Default set as True, indicating that the input dataframe containing the certainty distributions
    dist_stats takes a cert_dist dataframe as an input. Further, there is a fixed parameter, contains_test, that assumes the cert_dist dataframe has data whose predictive_status is unknown and labeled as 'test'. If comparing with a baseline model, be sure to include contains_test=False, in order to avoid empty columns.
    
    Returns
    -------
    dist_stats: dict()
        Dist_stats is a dictionary that displays relevant statistics in the form of global and local, label respective certainties between TP and FPs, and their corresponding distributions.
        Additionally, gathers empirical competence and component competence scores, and the Mann Whitney U test scores corresponding to each distribution tag and predictive status.
    """
    df = kwargs['cert_dist']
    pair_compare = [('TP','FP'), ('TP','test'),('FP','test')]
    dist_stats = dict()
    for name in df.prediction.unique():
        inner_stat = dict()
        inner_stat.__setitem__('stats',df[['certainty_score','predictive_status']][df.prediction==name].groupby('predictive_status').agg({'certainty_score':['count','min','max','median','mean','std']}))
        if contains_test:
            #inner_stat.__setitem__('kruskal-wallis', stats.kruskal(*[df.certainty_score[(df.predictive_status==kind) &(df.prediction==name) ] for kind in df.predictive_status.unique()]))
            inner_stat.__setitem__('mann-whitney',pd.concat([pd.DataFrame(stats.mannwhitneyu(df.certainty_score[(df.predictive_status==pair[0]) &(df.prediction==name)],df.certainty_score[(df.predictive_status==pair[1]) &(df.prediction==name)]), index=['statistics','p-value'],columns=[pair]) 
                                                             if (len(df.certainty_score[(df.predictive_status==pair[0]) &(df.prediction==name)]>0) & len(df.certainty_score[(df.predictive_status==pair[1]) &(df.prediction==name)])>0)
                                                             else pd.DataFrame({pair:['','']},index=['statistics','p-value'])
                                                             for pair in pair_compare],axis=1))
        else:
            inner_stat.__setitem__('mann-whitney',pd.concat([pd.DataFrame(stats.mannwhitneyu(df.certainty_score[(df.predictive_status=='TP') &(df.prediction==name)],df.certainty_score[(df.predictive_status=='FP') &(df.prediction==name)]),
                                                                          index=['statistics','p-value'],columns=[('TP','FP')])
                                                            if (len(df.certainty_score[(df.predictive_status=='TP') &(df.prediction==name)]>0) & len(df.certainty_score[(df.predictive_status=='FP') &(df.prediction==name)])>0)
                                                            else pd.DataFrame({('TP','FP'):['','']},index=['statistics','p-value'])
                                                            ],axis=1))
        inner_stat.__setitem__('empirical competence', pd.DataFrame([empirical_competence(df[df.prediction==name])],index=['empirical competence'],columns=['name']))
        dist_stats.__setitem__(name,inner_stat)
    dist_stats.__setitem__('stats',df[['certainty_score','predictive_status']].groupby('predictive_status').agg({'certainty_score':['count','min','max','median','mean','std']}))
    dist_stats.__setitem__('empirical competence',empirical_competence(df))
    dist_stats.__setitem__('component competence',component_competence(df.prediction,df.classification_cat))
    if contains_test:
        dist_stats.__setitem__('mann-whitney', pd.concat([pd.DataFrame(stats.mannwhitneyu(df.certainty_score[(df.predictive_status==pair[0])],df.certainty_score[(df.predictive_status==pair[1])]),index=['statistics','p-value'],columns=[pair]) for pair in pair_compare]))
        #dist_stats.__setitem__('kruskal-wallis', stats.kruskal(*[df.certainty_score[df.predictive_status==kind] for kind in df.predictive_status.unique()]))
    else:
        dist_stats.__setitem__('mann-whitney',pd.concat([pd.DataFrame(stats.mannwhitneyu(df.certainty_score[(df.predictive_status=='TP') &(df.prediction==name)],df.certainty_score[(df.predictive_status=='FP') &(df.prediction==name)]),index=['statistics','p-value'],columns=[('TP','FP')])
                                                         if ((len(df.certainty_score[(df.predictive_status=='TP') &(df.prediction==name)])>0)& (len(df.certainty_score[(df.predictive_status=='FP') &(df.prediction==name)])>0))
                                                         else pd.DataFrame({('TP','FP'):['','']},index=['statistics','p-value'])
                                                         for pair in pair_compare]))
    return dist_stats


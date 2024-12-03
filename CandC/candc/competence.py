# competence.py
import torch
import numpy
import pandas as pd

from typing import Union

def component_competence(predicted_prob:Union[torch.Tensor,numpy.ndarray],observed_label:Union[torch.Tensor,numpy.ndarray]):
    """Returns the component competence (cc) scores.
    
    Parameters
    ----------
    predicted_prob : torch.Tensor | numpy.array
        Matrix or higher tensor that contains the pseudo-probability vector for each sample
    observed_labels : numpy.array
        A corresponding array that is used to select each samples observed label
        
    Returns
    -------
    cc: torch.Tensor| numpy.array
        The cc is a point or array of the difference(s) of the average probability estimates for the true label and the reciprocal of the total number of labels possible
    """
    if isinstance(predicted_prob,torch.Tensor):
        if len(predicted_prob.shape)==2:
            cc=torch.Tensor([predicted_prob[n][observed_label[n]] for n in range(predicted_prob.shape[0])])
            cc=cc.mean()-(1/predicted_prob.shape[1])
            cc=cc.numpy()
        elif len(predicted_prob.shape)==3:
            cc=[]
            for i in range(predicted_prob.shape[0]):
                temp=torch.Tensor([predicted_prob[i][n][observed_label[n]] for n in range(predicted_prob.shape[1])])
                temp=temp.mean()-1/predicted_prob.shape[2]
                cc.append(temp)
            cc=torch.Tensor(cc).numpy()
    else:
        cc = np.array([predicted_prob[i][observed_label[i]] for i in range(len(predicted_prob))])
        cc = cc.mean()-1/predicted_prob.shape[0] #(replaced len(predicted_prob[0]))
    return cc
   
def empirical_competence(df:pd.DataFrame):
    """Returns the empirical competence score (comp_score) of a corresponding distribution of certainty scores.
    
    Parameters
    ----------
    df: pandas.DataFrame
        Data frame must contain the certainty score distribution columns : {'classification_cat', 'prediction', 'certainty_score', 'predictive_status'}
    
    Returns
    -------
    comp_score : float
        Empirical competence scores assess the ability of a model to assign True Positives with high certainty and False Positives with low certainty.
        It is related to the Mann-Whitney U score, in that high absolute competence scores indicate the distribution between True and False Positives is likely to be different.
    """
    comp = df.certainty_score[df.predictive_status=='TP'].sum()
    incomp = df.certainty_score[df.predictive_status=='FP'].sum()
    comp_score = comp-incomp
    norm_factor = len(df[(df.predictive_status=='TP') | (df.predictive_status=='FP')]) 
    comp_score = comp_score/norm_factor if norm_factor >0 else comp_score
    return comp_score
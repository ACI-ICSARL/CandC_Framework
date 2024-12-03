# cost.py
import torch
from math import asin,pi,tan,sin,cos
import numpy as np
from typing import Union, Optional

def chi(idx:int,*args)->float:
    """ Returns the 'chi' score described in Berenbeim et al 2023, which is the product of certainty scores by the column of the corresponding complete certainty matrix
    
    Parameters
    ----------
    idx: int
        Describes fixed index of an array/probability vector
    args: numpy.array | list of float type
        1-d array consisting of certainty scores
    
    Returns
    -------
    chi: float
    """
    chi=1
    for index in range(len(args)):
        if index !=idx:
            chi*=(args[idx]-args[index])
    return chi

def vartheta(chi:float)->float:
    """ Returns the corresponding angle from the Riemannian projection on the chi score
    
    Parameters
    ----------
    chi: float
        Can be any real value or np.Inf value
    
    Returns
    -------
    theta: float
        float value computed by applying arcsin to the fraction of 1- chi**2 divided by 1+chi**2
    
    """
    return asin((1-chi**2)/(1+chi**2))

def cost_fn(predictions:Union[torch.Tensor,np.array],labels:Union[np.ndarray,list,torch.Tensor],custom_penalty=None)->float:
    """ Returns the total cost of errors made by applying a custom penalty function or otherwise fixed penalty function to an array of predictions consisting of pseudo-probability vectors
    Parameters
    ----------
    predictions: torch.Tensor | np.array
        Intended to be a 2-d matrix whose rows are samples and columns correspond to a pseudo-probability vector
    labels: np.array | list
        Corresponds to the true labels of each of the samples in predictions, used to determined if a prediction is to be penalized.
    custom_penalty : None| function taking float argument
        Defaults to None, but otherwise can substitute for a lambda expression on one float argument.

    Returns
    -------
    cost: float
        Computed as the sum of debits by the predictions made and the true labels.
    """
    cost=0.0
    if type(predictions)==torch.Tensor:
        predictions=predictions.numpy()
    if not isinstance(custom_penalty,None):
        penalty= custom_penalty
    else:
        penalty = lambda t: (pi/2-t)
    debit = lambda p,t : penalty(vartheta(chi(p.argmax(),*p)))*(p.argmax()!=t)
    for n in range(len(labels)):
        cost+=debit(predictions[n],labels[n])
    return cost
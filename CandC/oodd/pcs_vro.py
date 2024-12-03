# pcs_vro.py

import torch
import numpy as np
from math import log

def _get_mode_np(x):
    counts= np.unique(x)
    mode = np.amax(counts)
    return mode

def _get_mode_torch(x):
    _,idx= x.max(dim=1)
    idx = idx.flatten().numpy()
    return idx, _get_mode_np(idx)

def _correct_log_product(x):
    return 0 if x == 0 else x*log(x)

def predictive_entropy(predictive_probabilities):
    if type(predictive_probabilities)==torch.Tensor:
        predictive_probabilities=predictive_probabilities.numpy()
#        means = predictive_probabilities.mean(0)
#        return -(means*means.log()).sum()
#    else:
    means = np.mean(predictive_probabilities,axis=0)
    return np.sum(-np.vectorize(_correct_log_product)(means)) #*np.log(means))

def mutual_information_prediction(predictive_probabilities):
    pe=predictive_entropy(predictive_probabilities)
    if type(predictive_probabilities)==torch.Tensor:
        predictive_probabilities=predictive_probabilities.numpy()
        #ae=((predictive_probabilities*predictive_probabilities.log()).sum(1)).mean(0)
    
    ae = np.mean(np.sum(np.vectorize(_correct_log_product)(predictive_probabilities),axis=1))#(np.sum(predictive_probabilities*np.log(predictive_probabilities),axis=1)))
    return pe+ae
    
def vr(predictions_on_input:torch.Tensor):
    mode= predictions_on_input.mode(0)[0]
    return 1-(torch.vmap(lambda x: (mode==x))(predictions_on_input)).sum(0)/predictions_on_input.shape[0]

def vro(original_prediction,predictions_on_input):
    vro =(lambda x,y: 1-(x==y).sum(0)/y.shape[0])
    return vro(original_prediction,predictions_on_input)

def pcs_vro_fit(indata_pcs,indata_vro,tpr_threshold):
    """
    Finds the HPCS,LPCS,HVRO,LVRO so that tpr_threshold of the data is accepted, with the remaining being classified as unknown, and anything else being rejected
    """
    # We want to calibrate HPCS and LVRO so that we accept the tpr_threshold; since the samples may not be perfectly correlated, this will involve some trickery
    HPCS = indata_pcs.quantile(1-tpr_threshold,interpolation="higher").cpu()
    LVRO = indata_vro.quantile(tpr_threshold,interpolation="lower").cpu()
    pcs_sort,pcs_sort_idx = indata_pcs.sort(descending=True)
    accept_region = torch.vmap(lambda x,y: (x>=HPCS) & (y<=LVRO))
    current_HPCS_idx = (pcs_sort==HPCS).nonzero()[0,0].item()
    try:
        while (( indata_pcs.reshape(-1) >= HPCS) & ( indata_vro.reshape(-1)<=LVRO) ).sum() < (indata_pcs.reshape(-1).shape[-1]* tpr_threshold):
            current_HPCS_idx+=1
            HPCS=pcs_sort[current_HPCS_idx][0,0].item().cpu()
    except:
        HPCS=indata_pcs.min().cpu()
    # We will categorically reject any sample if the precision score is worse than what we observe in data and has higher vro than what we observe in distribution
    LPCS = indata_pcs.min().cpu()#indata_pcs.quantile(tpr_threshold,interpolation="lower").cpu()
    HVRO = indata_vro.max().cpu()
    
    
    return HPCS,LPCS,HVRO,LVRO

def pcs_vro_test(pcs,vro,HPCS,LPCS,HVRO,LVRO): 
    #HPCS=.7,LPCS=.4,HVRO=.6,LVRO=.4):
    """ Applies the pcs_vro test, with parameters determined by the pcs_vro_fit test above. In particular, we have written the test to accept as in-distribution entries with high predictive confidence/high certainty scores, and low variation rate scores, and reject those with low confidence/certainty and high variation, outputting 0 if in-distribtuion, .5 if unknown, and 1 if reject. 

    In order to retain the use of 
    IN DEVELOPMENT---- Since we're attempting to use a Tensor in some data-dependent control flow, and pytorch  doesn't support that yet, we need to implement a decidedly less efficient way of getting our results.
    """
    acc = lambda x,y: (x>=HPCS) & (y<=LVRO)
    rej = lambda x,y: (x<LPCS) & (y>HVRO)
    app_acc = torch.vmap(acc)
    app_rej = torch.vmap(rej)
    out_acc= app_acc(pcs,vro)
    out_rej = app_rej(pcs,vro)
    out_acc=out_acc.float()
    out_rej=out_rej.float()
    out = 0.5*(out_acc+1.0-out_rej)
    return 1.-out
    #accept = lambda x,y: 0 if (x>=HPCS) & (y<=LVRO) else 0.5
    #reject = lambda x,y: 0.5 if (x<LPCS) & (y>HVRO) else 0
    #out = np.vectorize(lambda x,y: accept(x,y) + reject(x,y))
    #return torch.Tensor(out(pcs.numpy(),vro.numpy())).cpu()

def pcs_vro_test_summary(pcs,vro,HPCS,LPCS,HVRO,LVRO): 
    #HPCS=.7,LPCS=.4,HVRO=.6,LVRO=.4):
    """ Applies the pcs_vro test, with parameters determined by the pcs_vro_fit test above. In particular, we have written the test to accept as in-distribution entries with high predictive confidence/high certainty scores, and low variation rate scores, and reject those with low confidence/certainty and high variation, with the remaining being unknown. 
    """
    accept_region = torch.vmap(lambda x,y: (x>=HPCS) & (y<=LVRO) )
    reject_region = torch.vmap(lambda x,y: (x<LPCS) & (y>HVRO))
    accepts = accept_region(pcs.reshape(-1),vro.reshape(-1)).sum()/pcs.reshape(-1).shape[-1]
    rejects = reject_region(pcs.reshape(-1),vro.reshape(-1)).sum()/pcs.reshape(-1).shape[-1]
    return dict({'accepts':accepts, 'rejects':rejects, 'unknown':1-(accepts+rejects)})
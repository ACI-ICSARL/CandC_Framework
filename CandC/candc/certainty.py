# certainty.py

import torch
import numpy as np

def component_certainty(probvec:torch.Tensor)->torch.Tensor:
    """
    Compute the component certainty tensor
    Parameters
    ----------------
    :probvec: Given a tensor, compute the component certainty tensor; default is to suppose this is (1,0)-tensor
    """
    return torch.ones(probvec.shape)*probvec.unsqueeze(-1)-probvec*torch.ones(probvec.shape).t()

def complete_component_certainty(comp_cert:torch.Tensor)->torch.Tensor:
    """
    Compute the complete component certainty tensor, which adds the identity matrix for each corresponding component certainty matrix
    Parameters
    ----------------
    :comp_cert: component certainty tensor
    """    
    return comp_cert+ torch.eye(comp_cert.shape[0])

def get_certainty(predictions)->(torch.Tensor,torch.Tensor,torch.Tensor):
    """ Returns the predicted labels (predicted_labels), certainty matrices (cert), and certainty scores (cert_score) from input predictions.
    
    Parameters
    ----------
    predictions : torch.Tensor
        * Must be matrix of whose rows are samples, and whose columns are corresponding logits or pseudoprobabilities
    
    Returns
    -------
    predicted_labels: torch.Tensor
        Returns the predicted label /category treating sample input as respective row information
    cert: torch.Tensor 
        Returns collection of matrix objects corresponding to the certainty matrix of a sample input's pseudo-probability vector.
    cert_score:  torch.Tensor
        Returns the scalar certainty score of each sample, indicating the strength of the highest value in a pseudo-probability vector relative to the next nearest option.
"""
    preds=predictions.detach().cpu()
    try:
        cco = torch.vmap(component_certainty,in_dims=0)(preds)
        ccc = torch.vmap(complete_component_certainty,in_dims=0)(cco)
        cert = ccc
    except:
        ccc_list = []
        for i in range(predictions.shape[0]):
            ccc_list.append(complete_component_certainty(component_certainty(preds[i])))
        ccc= torch.stack(ccc_list)
    predicted_label = preds.max(-1).indices
    cert_score = ccc[torch.arange(ccc.size(0)).unsqueeze(1),predicted_label.unsqueeze(1)].min(-1).values
    return cert,cert_score,predicted_label

def get_batch_certainties(predictions:torch.Tensor)->torch.Tensor:
    """
    Function to compute certainties for batches of predictions
    Params
    --------------------------------
    :predictions: torch.Tensor object whose shape is a triple
    """
    try:
        return torch.vmap(get_certainty,in_dims=0)(predictions)
    except RuntimeError:
        cert_list=[]
        cert_score_list=[]
        label_list=[]
        for i in range(predictions.shape[0]):
            c,cs,pl= get_certainty(predictions[i])
            cert_list.append(c)
            cert_score_list.append(cs)
            label_list.append(pl)
        return torch.stack(cert_list), torch.stack(cert_score_list),torch.stack(label_list)

def get_upper_cert_as_vec(cert:torch.Tensor)->torch.Tensor:
    """ A dimension reduction of the certainty tensor into a vector from the upper-triangular component of the tensor.
    Parameters
    -----------------
    :cert: torch.Tensor, a component or complete component certainty matrix.
    """
    indices = torch.triu_indices(cert.shape[0],cert.shape[1],offset=1)
    vec = cert[indices[0],indices[1]].flatten()
    return vec

def get_batch_upper_cert_as_vec(cert:torch.Tensor)->torch.Tensor:
    """
    Function to transform batches of certainties into a (1,1)-tensor where each row entry corresponds to the uppertriangular matrix of the corresponding certainty matrix
    Parameters
    --------------------
    :cert: torch.Tensor, tensor of certainties whose shape is a triple
    """
    try:
        return torch.vmap(lambda x: torch.vmap(get_upper_cert_as_vec,in_dims=0,out_dims=0)(x),in_dims=0,out_dims=0)(cert)
    except RuntimeError:
        print('Raised RunTime Error, moving to cert list')
        try:
            cert_list=[]
            for i in range(cert.shape[0]):
                cert_list.append(get_upper_cert_as_vec(cert[i]))
            return torch.stack(cert_list)
        except:
            print('Too large, gathering final 10')
            cert_list=[]
            for i in range(cert.shape[0]-10,cert.shape[0]):
                cert_list.append(get_upper_cert_as_vec(cert[i]))
            return torch.stack(cert_list)

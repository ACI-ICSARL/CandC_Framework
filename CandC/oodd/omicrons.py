# omicrons.py

import torch
import torch.nn as nn
import math 
from ..candc.certainty import ( get_certainty,
    get_upper_cert_as_vec,
)
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from typing import Optional
import numpy as np
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_precision_recall_curve,
    binary_roc,
)
from torchmetrics.utilities.compute import auc

def omicron_fn(new_input:torch.Tensor,sample:torch.Tensor):
    """ Computes the omicron value from an input tensor relative to a sample. Alternate norms may be considered. We default to the Frobenius norm."""
    if len(new_input.shape)==1:
        new_input=new_input.reshape(1,-1)
    try:
        omicrons = torch.vmap(lambda arg1: (torch.vmap(lambda arg2: torch.norm(arg1-arg2),in_dims=0))(sample),in_dims=0)(new_input).mean(1)
    except RuntimeError as re:
        print("Runtime error raised!\n{}\n Trying to subsample the sample input certainties".format(re))
        Alt = min(sample.shape[0],int(1e8//sample.shape[1]))
        n = min(200,Alt)
        print("We will sample {} certainties".format(n))
        try:
            rand_columns = torch.randperm(sample.shape[0])[:n].long()
            omicrons = torch.vmap(lambda arg1: (torch.vmap(lambda arg2: torch.norm(arg1-arg2),in_dims=0))(sample[rand_columns]),in_dims=0)(new_input).mean(1)
        except Exception as E:
            print("Still running into error {}\n will have to proceed iteratively (outer loop only)".format(E))
            omicrons_list =[]
            try:
                for j in tqdm.tqdm(range(new_input.shape[0]),desc="Subsampling omicrons"):
                    omicrons_list.append((torch.vmap(lambda arg1: (torch.vmap(lambda arg2: torch.norm(arg1-arg2),in_dims=0))(sample[rand_columns]),in_dims=0)(new_input[j])).mean())
                omicrons = torch.cat(omicrons_list)
            except Exception as E:
                print(E)
                print("Proceeding iteratively, inner and outer loop")
                for j in tqdm.tqdm(range(new_input.shape[0]),desc="Subsampling omicrons"):
                    rand_columns = torch.randperm(sample.shape[0])[:n].long()
                    omicrons_inner_list=[]
                    for i in rand_columns:
                        omicrons_inner_list.append(torch.norm(new_input[j]-sample[i]))
                    omicrons_list.append(torch.Tensor(omicrons_inner_list))
                omicrons= torch.stack(omicrons_list).mean(1)
    except IndexError:
        if len(sample)<1:
            I = torch.eye(new_input.shape[1])
            indices = torch.triu_indices(I.shape[0],I.shape[1],offset=1)
            Ivec = I[indices[0],indices[1]].flatten()
            omicrons = torch.vmap(lambda arg1: (torch.vmap(lambda arg2: torch.norm(arg1-arg2),in_dims=0))(Ivec),in_dims=0)(new_input).mean(1)
    if omicrons.shape[0]==new_input.shape[0]:
        return omicrons
    else:
        print('Omicron returned does not match input size, returning as tuple with omicrons as first argument and original input as second argument.')
        return omicrons,new_input
        
def _random_selection(Ts:torch.Tensor.shape,num_samples:int,max_sample_factor=50,weights:Optional[torch.Tensor]=None):
    """ Function to randomly select from an input tensor T without replacement
    Parameters
    -------------------------
    :Ts: Shape of the underlying Tensor to sample from
    :num_samples: number of samples to draw
    :max_sample_factor: indicates the upper bound of samples to draw
    :weights: optional torch.Tensor to use to weight the sample draw
    """
    max_size = max_sample_factor*Ts[1]
    if num_samples > Ts[0]:
        num_samples = Ts[0]
    if num_samples >max_size:
        num_samples = max_size
    if weights is None:
        weights = torch.ones(Ts[0])/Ts[0]
    idx = weights.multinomial(num_samples=num_samples,replacement=False)
    return idx

def _omicron_fn_large_inner(r:torch.Tensor,T:torch.Tensor)->torch.Tensor:
    """ Return omicron for respective row vector against sample matrix of flattened certainties
    Parameters
    -------------------
    :r: torch.Tensor representing the row vector to use when computing the omicron
    :T: torch.Tensor containing the certainties to use when computing the omicron
    """
    o=(torch.vmap(torch.norm)(r.reshape(1,-1)-T)).mean().cpu().item()
    return o

def omicron_fn_LARGE(test_sample,in_sample,same_sample=False,device='cpu',MAX_SAMPLE_FACTOR=30,weighted_index:Optional[torch.Tensor]=None)->torch.Tensor:
    """ When input certainty tensors are of sufficiently large dimension, use omicron_fn_LARGE, which samples the omicron statistics bounded above by the dimension of the certainty 
    modulo a fixed constant. Default weighted index is uniform
    :test_sample: torch.Tensor containing the certainties from an unknown, or novel input source, or reference sample
    :in_sample: torch.Tensor containing the certainties drawn from the reference sample
    :same_sample: bool, default=False, indicates that the test and in_sample draws are from the same reference sample
    :device: torch.device indicating what cuda device to use, defaults as 'cpu'
    :MAX_SAMPLE_FACTOR: int indicating the maximum sample size to use for computing the omicron statistic
    :weighted_index: defaults to None

    Return
    ----------
    torch.Tensor of omicrons
    """
    olist=[]
    print('Running omicron_fn_LARGE due to large sample size')
    R = test_sample.detach().clone().to(device)
    T = in_sample.detach().clone().to(device)
    if weighted_index==None and same_sample:
        weighted_index =torch.ones(in_sample.shape[0])-1/(in_sample.shape[0]-1)
    elif  weighted_index==None:
        weighted_index =torch.ones(T.shape[0]-1)/(T.shape[0]-1)
    if same_sample:
        for i in tqdm(range(test_sample.shape[0])):
            r = R[i]
            Tr= torch.cat((T[:i][:],T[(i+1):][:]))
            olist.append(_omicron_fn_large_inner(r,Tr[_random_selection( Ts=Tr.shape,num_samples=30*int(math.sqrt(Tr.shape[1])),weights=weighted_index,max_sample_factor=MAX_SAMPLE_FACTOR)]))
            del r
            del Tr            
    else:
        for i in tqdm(range(test_sample.shape[0])):
            olist.append(omicron_fn_large_inner(test_sample[i].detach().clone().to(device),
                                                in_sample[_random_selection(in_sample.shape,in_sample.shape[1])],
                                                weighted_index,
                                                device,
                                                MAX_SAMPLE_FACTOR))
    o=torch.Tensor(olist)
    del olist   
    return o


def make_test_omicrons_input(unknown_sample:torch.Tensor,omicrons_d:dict,**params)->torch.Tensor:
    """ Function to produce omicrons from a test sample and a pre-formed dictionary of omicrons for an in-sample population,

    Parameters
    --------------------------
    :unknown_sample: torch.Tensor, sample of certainties
    :omicrons_d: dict object derived from an Omicron_Data class object
    :params: additional parameters consisting of the number of classes, and predictive status to condition on     
    """
    num_classes=params['n_class'] 
    predictive_status = params.get('predictive_status','Global')
    stacklist=[]
    test_sample = unknown_sample.clone()
    test_sample = test_sample.detach().to('cpu')
    #print("The shape of the test sample is {}. The first sample of the test sample has shape {}".format(test_sample.shape,test_sample[0].shape))
    try:
        for sample in range(test_sample.shape[0]):
            certainty,cs,cat = get_certainty(torch.unsqueeze(test_sample[sample],0))
            certainty=torch.squeeze(certainty,0)
            cat = cat.item()
            certainty = get_upper_cert_as_vec(certainty)
            if (cat in omicrons_d.keys()):
                input_sample = omicrons_d[cat][predictive_status]['certainty_sample']
                if input_sample is not None: # & (type(certainty)!=None): 
                    try:
                        omicron = omicron_fn(certainty,input_sample)
                        omicron = omicron.reshape(1)
                        ec = torch.Tensor([omicrons_d[cat]['empirical_competence']])
                        om_ec_interaction = omicron *ec
                        stacklist.append(torch.cat([omicron,ec,om_ec_interaction]))
                    except AttributeError as e:
                        if (input_sample) is None:
                            omicron = cs.resize(1)
                            ec = torch.Tensor([0])
                            om_ec_interaction= omicron*ec
                            stacklist.append(torch.cat([omicron,ec,om_ec_interaction]))
                        else:
                            raise AttributeError
                    except ValueError as VE:
                        print("Error {} on sample {} for cat {} with corresponing certainty {} and omicrons_data entry {}".format(VE,sample,cat,certainty,omicrons_d[cat][predictive_status]))
            else:
                omicron = cs.resize(1)
                ec = torch.Tensor([0])
                om_ec_interaction = omicron*ec
                stacklist.append(torch.cat([omicron,ec,om_ec_interaction])) 
    except Exception as E:
        print(E)
        for sample in range(test_sample.shape[0]):
            certainty,cs,cat = get_certainty(test_sample[sample])
            cat = cat.item()
            certainty = get_upper_cert_as_vec(torch.squeeze(certainty,0))
            if (cat in omicrons_d.keys()):
                input_sample = omicrons_d[cat][predictive_status]['certainty_sample']
                if input_sample is not None: # & (type(certainty)!=None): 
                    try:
                        omicron = omicron_fn(certainty,input_sample)
                        omicron = omicron.reshape(1)
                        ec = torch.Tensor([omicrons_d[cat]['empirical_competence']])
                        om_ec_interaction = omicron *ec
                        stacklist.append(torch.cat([omicron,ec,om_ec_interaction]))
                    except AttributeError as e:
                        if (input_sample) is None:
                            omicron = cs.resize(1)
                            ec = torch.Tensor([0])
                            om_ec_interaction= omicron*ec
                            stacklist.append(torch.cat([omicron,ec,om_ec_interaction]))
                        else:
                            raise AttributeError
                    except ValueError as VE:
                        print("Error {} on sample {} for cat {} with corresponing certainty {} and omicrons_data entry {}".format(VE,sample,cat,certainty,omicrons_d[cat][predictive_status]))
            else:
                omicron = cs.resize(1)
                ec = torch.Tensor([0])
                om_ec_interaction = omicron*ec
                stacklist.append(torch.cat([omicron,ec,om_ec_interaction]))
    finally:
        test_input = torch.stack(stacklist)
        return test_input
    
def _make_internal_omicron_test_inputs(omicron_d:dict,**params):
    """
    Run to concatenate the omicron input data from the omicron_Data by category/key
    """
    pred_stat = params['predictive status'] if 'predictive status' in params.keys() else 'Global'
    N_class = params['n_class']
    stacklist = []
    for cat in omicron_d.keys():
        omicrons = omicron_d[cat][pred_stat]['omicrons'].clone().detach()
        omicrons=omicrons.reshape(1,-1)
        ec = omicron_d[cat]['empirical_competence']*torch.ones(omicrons.shape)
        omicron_ec_interaction = omicrons*ec
        stacklist.append(torch.cat([omicrons,ec,omicron_ec_interaction]).t())
    internal_input = torch.cat(stacklist)
    return internal_input
    
def _get_internal_omicron_test_labels(omicron_d):
    """ The internal omicron labels technically refer to the internal tests intended classification of TP or FP, not in-distribution or out-of-distribution. we determine a priori that all input data that is 'novel' belongs to a novel category when applying the internal omicron test.
    """
    return labels

def omicron_test_results(test_sample,omicron_d,internal_test,tpr_threshold=.95,**testparams):
    """
    Given a test_sample of certainties, perform an internal omicron test, and an external test that is similarly calibrated.
    Will (1) compute the external omicrons from the omicron_d[cat]['certainty_sample'], and aggregate into an input tensor
        (2) apply the internal_test_model to the test_omicron inputs get the scores for the external test
        (3) collate the omicrons from the omicron_d
        (4) apply the internal test model to the collated omicrons
        (5) record the internal test scores and aggegate them
        (6) gather the internal test results (accept/rejects) for the external data
        (7) gather external test data
    """

    # Gathering the test omicron input data for the OODD/unknown input data
    test_omicrons_inputs = make_test_omicrons_input(test_sample,omicron_d,**testparams)
    # Applying the internal test to the external data
    internal_scores_test_sample = internal_test.apply(test_omicrons_inputs.detach().numpy())
    # For comparison, getting the internal omicron inputs
    internal_omicron_inputs = _make_internal_omicron_test_inputs(omicron_d,**testparams)
    # Applying the internal test to extract the internal classification
    internal_omicron_scores= internal_test.apply(internal_omicron_inputs.clone().detach().numpy())
    # Collecting the combined scores
    internal_test_omicron_scores = torch.cat([torch.Tensor(internal_scores_test_sample),torch.Tensor(internal_omicron_scores)])
    # Collecting the combined labels
    internal_test_status = torch.cat([torch.zeros(test_sample.shape[0]),torch.Tensor(internal_test.label).flatten()])
    #internal_test_status = torch.cat([torch.zeros(internal_test_scores.shape[0]),torch.ones(internal_test_omicron_scores.shape[0])])
    test_dict= dict({'external omicrons' :test_omicrons_inputs[:,0],
                     'ec:cat': test_omicrons_inputs[:,1],
                     'internal test scores':internal_test_omicron_scores,
                     'internal test status':internal_test_status,
                    })
    test_dict.update(perform_external_test(test_sample,omicron_d,**testparams))
    test_dict.update(_oodd_omicrons(omicron_test=test_dict,tpr_threshold=tpr_threshold))
    return test_dict

def _oodd_omicrons(omicron_test,tpr_threshold=.95):
    ytrue_internal = omicron_test['internal test status']
    yscores_internal = omicron_test['internal test scores']
    print("INTERNAL OMICRON TEST\n----------------------------")
    internal_results = __oodd_metrics_internal(scores=yscores_internal, labels=ytrue_internal, tpr_threshold=tpr_threshold)
    ytrue_external =  torch.Tensor(omicron_test['external_test'].label)
    yscores_external = omicron_test['external test scores']
    print("EXTERNAL OMICRON TEST\n----------------------------")
    external_results = __oodd_metrics_internal(scores=yscores_external, labels=ytrue_external, tpr_threshold=tpr_threshold)
    return dict({'OmTestInt':internal_results,
                 'OmTestExt':external_results,
                })

def perform_external_test(test_sample,omicron_d,**params):
    """ Given a test sample of certainties, we want to:
    1) Set up the inputs for the external test
    2) train an external classifier with in-distribution and unknown distribution data
    3) set classifier fitness threshold to tpr_threshold for in-distribution data
    4) save the scores of the external test
    5) save the results of the external test
    """
    external_test_inputs,external_test_outputs=make_external_test_inputs_and_outputs(test_sample,omicron_d,**params)
    external_params = make_external_params(external_test_inputs,external_test_outputs,**params)
    external_test = log_omicron_test(**external_params)
    external_test.fit()
    external_test_scores = external_test.apply(external_test.input)
    #external_test_results = external_test.apply_test(external_test.input)
    return dict({'external_test_inputs':external_test_inputs,
                 'external_test_outputs':external_test_outputs,
                 'external_test':external_test, 
                 'external test scores':external_test_scores,
                 #'external test results':external_test_results,
                })

def make_external_test_inputs_and_outputs(test_sample,omicron_d,**params):
    """ Given a test sample of certainties and corresponding dictionary of omicrons, form the input object for the log omicron test object, and the classification vector for the outputs.
    """
    novel_inputs = make_test_omicrons_input(test_sample,omicron_d,**params)
    known_inputs = _make_internal_omicron_test_inputs(omicron_d,**params)
    inputs = torch.cat([novel_inputs,known_inputs])
    outputs = make_test_outputs(novel_inputs.shape[0],known_inputs.shape[0])#test_sample,omicron_d,**params)
    if inputs.shape[0]==outputs.shape[0]:
        return inputs,outputs
    else:
        print("Shapes are mismatched input:{} != output:{}".format(inputs.shape[0],outputs.shape[0]))
        raise ValueError

def make_external_params(inputs,outputs,**params):
    external_params=dict()
    external_params.__setitem__('input', inputs)
    external_params.__setitem__('distribution_status',outputs)  # vector must be included and coded as 0/1
    external_params['tpr_threshold'] = params['tpr_threshold'] if 'tpr_threshold' in params.keys() else .95
    if 'logistic_params' in params.keys():
        external_params.__setitem__('logistic_params', params.get('logistic_params'))
    return external_params

def make_test_outputs(N_novel_inputs:int,N_known_inputs:int):
    """ make tensor labels with the ou
    """
    return torch.cat([torch.zeros(N_novel_inputs),torch.ones(N_known_inputs)])
    
def generate_omicron_test(omicron_data:dict,test_params:dict):
    """
    Given a dictionary of omicron data organized, generates and trains an instance of the log_omicron_test
    Parameters
    -------------------------------
    :omicron_data: dictionary of omicron data, derived from the Omicron Data class
    :test_params: dictionary of additional parameters for customizing the logistic omicron-competence model 
    """
    # We iterate through the cat keys of together the omicron inputs from omicron_data[cat]['Global']
    stacklist_input=[]
    catlist_label=[]
    num_classes = test_params['n_class']
    pred_class = test_params['predictive_status']
    cats = list(omicron_data.keys()) #[key for key in omicron_data.keys() if type(key)!=str]
    for cat,o_data in omicron_data.items():
        omicrons = o_data[pred_class]['omicrons'].reshape(1,-1)
        ec = o_data['empirical_competence']*torch.ones(size=omicrons.size())
        om_ec_interaction = omicrons *ec
        inner_list = [omicrons,ec,om_ec_interaction]
        stacklist_input.append(torch.cat(inner_list).t())
        catlist_label.append(o_data[pred_class]['predictive_statuses'])
    test_input = torch.cat(stacklist_input)
    distribution_status = torch.cat(catlist_label)
    if torch.bincount(distribution_status).shape==torch.Size([1]):
        print("Cannot properly balance the omicron test; all internal data is of the same predictive status\n Will regenerate with several 'spurious' FP derived using the tpr_threshold")
        stacklist_input=[]
        catlist_label=[]
        num_classes = test_params['n_class']
        pred_class = test_params['predictive_status']
        d= 1-test_params.get('tpr_threshold',.95)
        random_tensor = torch.empty(10).uniform_(-d, d)+test_params.get('tpr_threshold',.95)
        for cat in omicron_data.keys():
            omicrons = omicron_data[cat][pred_class]['omicrons'].reshape(1,-1)
            ec = omicron_data[cat]['empirical_competence']*torch.ones(size=omicrons.size())
            om_ec_interaction = omicrons *ec
            false_omicrons = torch.quantile(omicrons,random_tensor,interpolation="linear").reshape(1,-1)
            false_ec = omicron_data[cat]['empirical_competence']*torch.ones(size=false_omicrons.size())
            false_om_ec_interaction = false_omicrons * false_ec
            cat_sample =[omicrons,  
                         ec,
                         om_ec_interaction,
                        ]
            false_cat_sample = [false_omicrons,
                                false_ec,
                                false_om_ec_interaction,
                               ]
            stacklist_input.append(torch.cat(cat_sample).t())
            stacklist_input.append(torch.cat(false_cat_sample).t())
            catlist_label.append(omicron_data[cat][pred_class]['predictive_statuses'])
            catlist_label.append(torch.ones(10))
        test_input = torch.cat(stacklist_input)
        distribution_status = torch.cat(catlist_label)
    logistic_params=dict({'penalty' : test_params.get('penalty', 'l2'),
                          'tol': test_params.get('tol', 1e-4),
                          'C': test_params.get('C', 1.0),                         
                          'fit_intercept':test_params.get('fit_intercept', False),
                          'class_weight': test_params.get('class_weight',None),   
                          'solver': test_params.get('solver','newton-cholesky'),
                          'max_iter': test_params.get('max_iter',100),
                          'n_jobs': test_params.get('n_jobs',128),})
    params=dict({'input' : test_input,
                 'distribution_status' : distribution_status,
                 'tpr_threshold' : test_params.get('tpr_threshold',.95),
                 'logistic_params' : logistic_params,
                })
    print(params.keys())
    om_test=log_omicron_test(**params)
    om_test.fit() 
    return om_test


def __fpr_at_tpr(pred, target, tpr_threshold=0.95):
    """                                                                                                
    Calculate the False Positive Rate at a certain True Positive Rate                        
    :param pred: outlier scores                                                             
    :param target: target label                                                             
    :param k: cutoff value                                                                  
    :return:                                                                                           
    """
    # results will be sorted in reverse order                                                          
    fpr, tpr, _ = binary_roc(pred, target)
    idx = torch.searchsorted(tpr, tpr_threshold)
    print("The tpr tensor is {}".format(tpr))
    if idx == fpr.shape[0]:
        print(f"INTERNAL ONLY: With cut-off at {tpr_threshold}, \n fpr {fpr} \n tpr {tpr}, the corresponding idx is {idx},\n with fpr95tpr {fpr[idx-1]}")
        return fpr[idx - 1]
    print(f"INTERNAL ONLY: With cut-off at {tpr_threshold}, \n fpr {fpr} \n tpr {tpr}, the corresponding idx is {idx}, \nwith fpr95tpr {fpr[idx]}")
    return fpr[idx]

def __oodd_metrics_internal(scores,labels,tpr_threshold,device='cpu'):
    """ We want to gather ACC,OODD,AUROC, AUPR IN, AUPR OUT, and FPR@95TPR from our detector, given inputs the oodd test scores and labels (known in vs unknown)
    Note: outputs y will need to be from a combined dataset with our in and out of distribution data, with all out of distribution data given a negative label to reliably 
    class as unknown or out of distribution; when training the classifier above, the out-of-distribution data is then collapsed to 0 while all non-negative labels are 
    collapsed to 1, indicating in-distribution.

    Parameters
    -------------
    :scores: torch.Tensor containing the detector results
    :labels: torch.Tensor containing the in/out distribution labels 1/0 on input, which are inverted when computing
    OODD scores as in pytorch_ood repo
    :device: torch.device to use
    """
    scores=scores.to(device)
    labels=labels.to(device)
    scores, scores_idx = torch.sort(scores, stable=True)
    labels = labels[scores_idx]
    labels= labels.long()
    auroc = binary_auroc(scores, 1-labels)

    print("The scores for the omicron test are {}\n The labels are {}".format(scores,labels))
    
        # num_classes=None for binary                                                                  
    p, r, t = binary_precision_recall_curve(scores, 1-labels)
    aupr_in = auc(r, p)

    p, r, t = binary_precision_recall_curve(1-scores, labels)
    aupr_out = auc(r, p)

    
    if tpr_threshold>=1.0:
        print("Resetting the tpr_threshold, too high")
        tpr_threshold=.95
    
    fprname = "FPR"+str(tpr_threshold).split('.')[-1]+"TPR"
    
    fpr = __fpr_at_tpr(pred= scores,target=(1-labels),tpr_threshold=tpr_threshold)

    output= dict({
            "AUROC": auroc.cpu(),
            "AUPR-IN": aupr_in.cpu(),
            "AUPR-OUT": aupr_out.cpu(),
            fprname: fpr.cpu(),
        })
    print("Omicron Test Results \n{}".format(output))
    return output

class log_omicron_test():
    f"""
    Build a logistic regression test for out-of-distribution/FP detection. 
    Accordingly, set OOD data to 0, and in-distribution data to 1.
    However, following the implementation in pytorch_ood, we determine a TP for category 0, and FP for category 1. 
    """
    def __init__(self,**params):
        self.input = params['input'].clone().detach().numpy()
        self.n_input = self.input.shape[1]
        self.label = params['distribution_status'].reshape(-1).float().clone().detach().numpy()
        self.model = LogisticRegression(**params.get('logistic_params',
                                                     dict({'penalty' : 'l2',
                                                           'tol':  1e-4,
                                                           'C':  1.0,     
                                                           'fit_intercept': False,
                                                           'class_weight': None, 
                                                           'solver': 'newton-cholesky',
                                                           'max_iter': 100,
                                                           'n_jobs': 128,})))

        #LogisticOmicronModel(self.n_input)
        self.tpr_threshold = params['tpr_threshold'] if 'tpr_threshold' in params.keys() else .95
        self.test_threshold = 0

    def fit(self):
        self.model.fit(X=self.input,y=self.label)
        print("There are {} total items.\n There are {} in-distribution items.\n There are {} out-of-distribution/FP items".format(self.input.shape[0], (self.label>0).sum(), (self.label<1).sum()))
        print("The accuracy of the log omicron model after fitting is {}".format(self.model.score(X=self.input, y=self.label)))

    def apply(self,inputs,display=False):
        prob_vector=torch.Tensor(self.model.predict_proba(X=inputs))
        if display:
            print("The predicted probability vector is {}".format(prob_vector))
        return prob_vector[:,0]

    def apply_to_output(self,outputs,omicrons_d,display:bool,**params):
        return self.apply(inputs=make_test_omicrons_input(unknown_sample=outputs,omicrons_d=omicrons_d,**params),display=display)
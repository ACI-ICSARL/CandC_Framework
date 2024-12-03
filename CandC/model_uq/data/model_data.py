# model_data.py
import torch
import numpy as np
import pandas as pd
import hamiltorch
import os
import pickle
from torch import argmax
from torch.nn.functional import softmax
from typing import Union,Optional
from itertools import product

class Input_Data():
    
    def __init__(self,data:dict[str,Union[torch.Tensor,np.array,str,bool,torch.utils.data.DataLoader]]):
        """ Contains the necessary input data except for the model whose uncertainty we're examining
        with the C&C Framework.

        Parameters
        --------------
        :name: Optional string value, naming the Input/Model Data object
        :input_data_features': optional value consisting of inputs to the model object, ideally 
        a torch.Tensor object containing the input features
        :input_dataloader: alternatively, a torch.DataLoader object containing features and labels
        :classification_categories: dict[int,Union[str,int]] object that assigns integer key to string label names
        :labeled_data: bool, indicates if the input_data_features or input_dataloader contains data 
        with a label
        :input_data_labeled: the corresponding np.array or torch.Tensor object containing the 
        corresponding labels for data
        :classification_scheme: optional functional argument to be applied to group classification labels 
        and prediction together
        :safevalues: optional list of labels that turn the classification into a binary 
        classification problem, so that if a prediction is in the safevalues list, we label as 1 and 0
        otherwise
        """
        possible_attributes = ['name',
                              'input_data_features',
                              'input_dataloader',
                              'classification_categories', 
                              'labeled_data',
                              'input_data_labeled',
                              'classification_scheme',
                              'safevalues',]
        for key in possible_attributes:
            if key in data.keys():
                setattr(self,key,data[key])
            if getattr(self,'labeled_data',False):
                if not 'input_data_labeled' in self.to_dict().keys():
                    print("Filling in data labels from dataloader.")
                    if 'input_dataloader' in self.to_dict().keys():
                        self.collect_data_labels()

    def set_name(self,name):
        """Set/Replce the name attribute"""
        setattr(self,'name',name)
        
    def set_input_data_features(self,input_data_features):
        """Set/Replace the input data features"""
        setattr(self,'input_data_features',input_data_features)

    def set_input_dataloader(self,input_dataloader):
        """ Set/Replace the input dataloader if detected"""
        setattr(self,'input_dataloader',input_data_loader)
    
    def set_classification_categories(self,classification_categories):
        """Set/Replace the classification_categories attribute"""
        setattr(self,'classification_categories',classification_categories)
    
    def set_input_labeled_data(self,input_labeled_data):
        """Set/Replace the input_labeled_data feature and boolean check"""
        setattr(self,'input_labeled_data',input_labeled_data)
        setattr(self,'labeled_data',True)
    
    def set_classification_scheme(self,classification_scheme):
        """Set/Replace the classification_scheme function if applicable"""
        setattr(self,'classification_scheme',classification_scheme)

    def make_full_gatherlist(self):
        classifications =['Global']+list(self.classification_categories.keys())
        status=['','TP','FP']
        return  product(classifications,status) 
    
    def to_dict(self):
        """ Convert instance attributes to a dictionary."""
        return self.__dict__
   
    def make_output(self,
                    model,
                    apply_softmax = True,
                    custom_weight_function = None,
                    is_bayesian = False,
                    device = torch.device('cpu'),
                    get_k_predictions : Optional[int]=None,
                    **additional_params):
        """Produceds the output_data from an input model and additional parameters.
        
        Parameters
        ----------
        :model: Although the present implementation of the C&C Framework relies on PyTorch 
        tensor objects, in principle we accept any model object prior to transforming the output
        into a corresponding torch.Tensor object provided the model object can interact with and transform
        the Input_Data object attributes into logit or pseudo probability vectors.
        :apply_softmax: bool indicating if we are to apply a softmax function to the output of the
        model applied to the input data.
        :custom_weight_function: Default to None. WARNING:Primarily intended to be used for any post-hoc
        transformation used to ensemble Bayesian NN data, any such post-hoc function may be composed here,
        including other neural network models treat the output of the moddel as an embedded feature.
        :is_bayesian: separate bool from custom_weight_function. Set to True if the output of the model is
        to be ensembled into a single predictor.
        :get
        :additional_params: Default assumptions with C&C Framework are any additional parameters are for
        parameters for running Hamiltonian Monte Carlo in hamiltorch. Separate logic to handle alternate
        forms of Bayesian sampling to be considered in future versions."""
        data=dict()
        if 'name' in self.to_dict().keys():
            data.__setitem__('name',self.name)
        try:
            if 'input_dataloader' in self.to_dict().keys():
                if is_bayesian:
                    output, log_probs = hamiltorch.predict_model(model,**additional_params)
                    prediction = argmax(output,dim=-1)
                    if custom_weight_function:
                        ensemble_output = custom_weight_function(output)
                    else:
                        ensemble_output = softmax(output,dim=-1).mean(0)
                    ensemble_prediction = argmax(ensemble_output,dim=-1)
                    data.__setitem__('ensemble_output',ensemble_output)
                    data.__setitem__('ensemble_prediction',ensemble_prediction)
                else:
                    logits = self.collect_model_outputs(model=model, 
                                                        device=device)
                    if apply_softmax:
                        output = softmax(logits,dim=-1)
                    else:
                        output = logits
                    prediction = argmax(output,dim=-1)
                data.__setitem__('output',output)
                data.__setitem__('prediction',prediction)
                
            elif 'input_data_features' in self.to_dict().keys():
                if is_bayesian:
                    # If bayes set to True, for now we suppose we are using hamiltorch
                    output, log_probs = hamiltorch.predict_model(model,**additional_params)
                    prediction = argmax(output,dim=-1)
                    if custom_weight_function:
                        ensemble_output = custom_weight_function(output)
                    else:
                        ensemble_output = softmax(output,dim=-1).mean(0)
                    ensemble_prediction = argmax(ensemble_output,dim=-1)
                    data.__setitem__('ensemble_output',ensemble_output)
                    data.__setitem__('ensemble_prediction',ensemble_prediction)
                else:
                    logits=model(self.input_data_features)
                    if apply_softmax:
                        output = softmax(logits,dim=-1)
                    else:
                        output = logits
                    prediction = argmax(output,dim=-1)
                data.__setitem__('output',output)
                data.__setitem__('prediction',prediction)
            if get_k_predictions:
                data.__setitem__('predictions',torch.topk(data.output,get_k_predictions,dim=-1))
                if is_bayesian:
                    data.__setitem__('ensemble_predictions',torch.topk(data.ensemble_output,get_k_predictions,dim=-1))
        except Exception as E:
            print("We have encountered the following error\n{}".format(E)) 
        finally:
            return data

    def collect_model_outputs(self,
                              model,
                              device: torch.device):
        model.to(device)
        model.eval()
        outputs_list = []
        with torch.no_grad():
            if self.labeled_data:
                for inputs,_ in self.input_dataloader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    outputs_list.append(outputs.cpu())
            else:
                for inputs in self.input_dataloader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    outputs_list.append(outputs.cpu())
        all_outputs = torch.cat(outputs_list,dim=0)
        return all_outputs

    def collect_data_labels(self):
        data_label_list = []
        for _,labels in self.input_dataloader:
            data_label_list.append(labels)
        data_labels = torch.cat(data_label_list)
        setattr(self,'input_data_labeled',data_labels)
    
    def save(self,address:str,filename:Optional[str]=None):
        """ Save a pickled representing the attributes of the Input_Data object"""
        dict_to_save = self.to_dict()
        if filename:
            input_data_name = filename+".pickle"
        else:
            input_data_name = "_input_data.pickle"
        with open(os.path.join(address,input_data_name),'wb') as handle:
            pickle.dump(dict_to_save, handle,protocol=pickle.HIGHEST_PROTOCOL)
        
    def load(self,address,name):
        """  Load in the saved representation of the Input_Data."""
        possible_attributes = ['name',
                              'input_data_features',
                              'input_dataloader',
                              'classification_categories', 
                              'labeled_data',
                              'input_data_labeled',
                              'classification_scheme',
                              'safevalues']
        if filename:
            input_data_name = filename+".pickle"
        else:
            input_data_name = "_input_data.pickle"
        with open(os.path.join(address,input_data_name),'rb') as handle:
            data = pickle.load(handle)
            for key in possible_attributes:
                if key in data.keys():
                    setattr(self,key,data[key])
            handle.close()

class Output_Data():
    def __init__(self,data:dict[str,Union[torch.Tensor,np.array,str,bool]]):
        """ Contains the necessary output data for processing the certainty and competence information."""
        # Initialize data attributes attributes
        # outputs, prediction, ensemble outputs, ensemble prediction
        possible_attributes = ['name',
                               'output',
                               'prediction',
                               'predictions',
                               'ensemble_output', 
                               'ensemble_prediction',
                              'ensemble_predictions']

        for key in possible_attributes:
            if key in data.keys():
                setattr(self,key,data[key])

    def set_output(self,output):
        """ Replace/set the model output information"""
        setattr(self,'output',output)

    def set_prediction(self,prediction):
        """Replace/set the model output prediction, specifically wrt to label or labels"""
        setattr(self,'prediction',prediction)

    def set_ensemble_output(self,ensemble_output):
        """Replace/set the ensemble models predicted outputs"""
        setattr(self,'ensemble_output',ensemble_output)

    def set_ensemble_prediction(self,ensemble_prediction):
        """Replace/set ensemble model prediction"""
        setattr(self,'ensemble_prediction',ensemble_prediction)
    
    def to_dict(self):
        """ Convert instance attributes to a dictionary."""
        return self.__dict__
        
    def save(self,address:str,filename:Optional[str]=None):
        """Save output_data object"""
        dict_to_save = self.to_dict()
        output_name = filename+'.pickle' if filename is not None else '_output_data.pickle'
        with open(os.path.join(address,filename),'wb') as handle:
            pickle.dump(dict_to_save, handle,protocol=pickle.HIGHEST_PROTOCOL)
        
    def load(self,address:str,filename:Optional[str]=None):
        """Load saved output_data object"""
        possible_attributes = ['name',
                               'output',
                               'prediction',
                               'predictions',
                               'ensemble_output', 
                               'ensemble_prediction',
                              'ensemble_predictions']
        if filename:
            output_data_name = filename+".pickle"
        else:
            output_data_name = "_output_data.pickle"
        with open(os.path.join(address,output_data_name),'rb') as handle:
            data = pickle.load(handle)
            for key in possible_attributes:
                if key in data.keys():
                    setattr(self,key,data[key])
            handle.close()

class Model_Data(Input_Data,Output_Data):
    def __init__(self,
                 model_data_address:Optional[str]=None,
                 model_data_filename:Optional[str]=None,
                 model=None,
                 data:Optional[dict[str,Union[torch.Tensor,np.array,str,bool]]]=None,
                 apply_softmax=True,
                 custom_weight_function=None,
                 is_bayesian=False,
                 **additional_params):
        """Joint object combining Input_Data and Output_Data objects to expedite their interaction"""
        try:
            if not model_data_address:
                print("Initializing with provided data and model.")
                Input_Data.__init__(self,data=data)
                Output_Data.__init__(self,
                                     data=self.make_output(model=model,
                                                           apply_softmax=apply_softmax,
                                                           custom_weight_function=custom_weight_function,
                                                           is_bayesian=is_bayesian,
                                                           **additional_params))
            else:
                print("Loading model_data object attributes.")
                self.load(address=model_data_address,name=model_data_filename)
        except Exception as E:
            print("While trying to load in model_data, raised the following error:{}".format(E))
            
    def to_dict(self):
        """ Convert instance attributes to a dictionary."""
        return self.__dict__
    
    def save(self,address:str,filename:Optional[str]=None):
        """Save combined Model Data as one object"""
        dict_to_save = self.to_dict()
        if filename:
            model_data_name = filename+".pickle"
        else:
            model_data_name = "_model_data.pickle"
        with open(os.path.join(address,model_data_name),'wb') as handle:
            pickle.dump(dict_to_save, handle,protocol=pickle.HIGHEST_PROTOCOL)
        
    def load(self,address:str,name:Optional[str]=None):
        """Load in stored model_data object"""
        possible_attributes = ['name',
                               'input_data_features',
                               'input_dataloader',
                               'classification_categories', 
                               'labeled_data',
                               'input_data_labeled',
                               'classification_scheme',
                               'safevalues',
                               'output',
                               'prediction',
                               'predictions',
                               'ensemble_output', 
                               'ensemble_prediction',
                               'ensemble_predictions']
        if name:
            model_data_name = name+".pickle"
        else:
            model_data_name = "_model_data.pickle"
        with open(os.path.join(address,model_data_name),'rb') as handle:
            data = pickle.load(handle)
            for key in possible_attributes:
                if key in data.keys():
                    setattr(self,key,data[key])
            handle.close()
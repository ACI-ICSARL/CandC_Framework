# certainties.py
import pandas as pd
import gc
import os
import pickle
from typing import Union,Optional

from .model_data import Output_Data,Model_Data
from .gather import *
from ...candc.certainty import *

class Certainties():
    def __init__(self,address:Optional[str]=None,name:Optional[str]=None):
        """ Class object storing the certainty, certainty score, and predictions as well as
        ensembled counterparts if needed, from a given model on arbitrary input (either labeled or unlabeled)
        """
        if not address:
            self.output = None
            self.certainty = None
            self.certainty_score = None
            self.predictions = None
            self.ensemble_output = None
            self.ensemble_certainty = None
            self.ensemble_certainty_score = None
            self.ensemble_predictions = None
            self.VR = None
            self.VRO = None
        else:
            try:
                self.load(address=address,name=name)
            except Exception as E:
                print("When attempting to load certainties object, we raise the following error:{}".format(E))

    def gather_certainties(self,
                           output_data:Union[Output_Data,Model_Data],
                           is_bayesian= False,
                           select_rate= .9,
                           select_sum = 20):
        """Fills in the Certainties attributes where appropriate based on the provided 
        output_data, and parameters:
        :is_bayesian: boolean indicating if the output_data should be treated as a Bayesian
        sample needing to be separately analyzed as an ensemble for the sake of certainties.
        :select_rate: default value .9; parameter used to select the tail end of samples for large
        batches of certainties when drawing from a Bayesian sample
        :select_sum: default value 20, parameter used to enforce the last numbers to select from a
        large Bayesian sample.
        """
        print("Now gathering certainties")
        if len(output_data.output.shape)==2:
            self.output=output_data.output.clone()
            certainty, certainty_score,predicted_label = get_certainty(self.output)
            print("Original certainty shape is {} from predictions shape {}".format(certainty.shape,output_data.output.shape))
            self.certainty=torch.vmap(get_upper_cert_as_vec)(certainty)
            if len(self.certainty.shape)>2:
                print("Reshaping certainty")
                self.certainty = self.certainty.reshape(self.certainty.reshape(self.certainty.shape[0],self.certainty.shape[1]))
        else:
            self.output=output_data.output.clone()
            certainty_list=[]
            certainty_score_list=[]
            predicted_label_list =[]
            SAMPLE_NUMS = self.output.shape[0]
            Start = max(int(select_rate* SAMPLE_NUMS),SAMPLE_NUMS-select_sum)
            print("There are many samples, we're going to start at sample {}".format(Start))
            sample = self.output.clone().detach().cpu()
            for n in tqdm(range(Start,SAMPLE_NUMS),desc="Gathering the certainty information"):
                sample_output = sample[n]
                certainty,certainty_score,predicted_label = get_certainty(sample_output)
                certainty_list.append( (torch.vmap(get_upper_cert_as_vec)(certainty)).unsqueeze(0))
                if n == Start:
                    print("The shape of the reshaped certainty is {}".format(certainty_list[0].shape))              
                    certainty_score_list.append(certainty_score)
                predicted_label_list.append(predicted_label)
            print("Stacking the certainties")
            print('parent process:', os.getppid())
            print('process id:', os.getpid())
            print("The number of certainty samples we gathered is {}\n The shape of each sample is {}".format(len(certainty_list),certainty_list[0].shape))
            #certainty_list = [get_upper_cert_as_vec()]
            certainty=torch.cat(certainty_list)
            print("Stacked certainty")
            certainty_score=torch.stack(certainty_score_list)
            print("Stacked certainty score")
            predicted_label=torch.stack(predicted_label_list)
            print("Stacked predictions")
            self.certainty=certainty    
        self.certainty_score = certainty_score
        self.predictions = predicted_label
        print("Finished gathering certainties.")
        gc.collect()
        
        if is_bayesian:
            print("Bayesian sample data detected")
            self.ensemble_output=output_data.ensemble_output.clone()
            certainty, certainty_score,predicted_label = get_certainty(self.ensemble_output)
            print("Ensemble certainty gathered")
            self.ensemble_certainty= torch.vmap(get_upper_cert_as_vec)(certainty)
            self.ensemble_certainty_score = certainty_score
            self.ensemble_predictions = predicted_label
            self.VR = vr(self.predictions)
            self.VRO = vro(self.ensemble_predictions.t(),self.predictions)

    def save_certainties_sample(self,sampled_certainties):
        setattr(self,"sampled_certainties",sampled_certainties)
    
    def to_dict(self):
        """ Convert instance attributes to a dictionary."""
        return self.__dict__
         
    def save(self,address:str,name:Optional[str]=None):
        """ Save a pickled representing the attributes of the Certainties object"""
        dict_to_save = self.to_dict()
        if name:
            certainties_name = name+".pickle"
        else:
            certainties_name = "_certainties.pickle"
        with open(os.path.join(address,certainties_name),'wb') as handle:
            pickle.dump(dict_to_save, handle,protocol=pickle.HIGHEST_PROTOCOL)

    def load(self,address:str,name:str):
        """ Load in a specified certainties object
        """
        try:
            #open
            if name:
                certainties_name = name+".pickle"
            else:
                certainties_name = "_certainties.pickle"
            with open(os.path.join(address,certainties_name),'rb') as file:
                certainties = pickle.load(file) 
            for name,attr in certainties.items():
                setattr(self,name,attr)
        except Exception as E:
            print("Failed to load due to the following error:\n {}".format(E))
        
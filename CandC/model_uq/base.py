import numpy as np 
import torch
import pandas as pd
import pickle
import gc
import os

from tqdm import tqdm
from typing import Union, Optional
from itertools import product
from torch.utils.data import DataLoader

from ..candc.certainty import (
    get_certainty,
    get_batch_certainties,
    get_upper_cert_as_vec,
    get_batch_upper_cert_as_vec,
)
from ..candc.competence import *
from ..certainty_stats.certainty_distribution import (
    find_certainty_dist_dataframe,
    dist_stats,
)
from ..certainty_stats.display import *
from ..oodd.mwu import  *
from ..oodd.omicrons import *
from ..oodd.pcs_vro import *
from ..oodd.oodd_tests import (
    oodd_test_and_metrics_external,
    oodd_metrics_internal,
)
from ..utils.regression import *
from .data import * 

class Model_UQ():
    r""" The MODEL_UQ_BASE object consists of the underlying model to which we want to acquire various uncertainty quantification data,  the addresses for conducting experiments,
    and a default tpr_threshold parameter used for the general model UQ object. 
    """
    
    def __init__(self,**params):
        """
        Parameters
        -----------------------
        :model: underlying model whose uncertainty we are quantifying
        :n_class: integer describing the total number of distinct classes an underlying discriminative model is examinging for the purposes of assessing competence
        
        (Optional) Parameters
        -----------------------
        :name: str, optional value to append to various objects to distinguish from other objects or model_uq instances.
        :device: torch.device, indicates which device model and data should be processed on
        :model_address: str, optional value indicating the path where the model is saved
        :data_address: str, optional value indicating the path where the model data should be stored
        :model_uq_address: str, optional value indicating the path where the model uq object should be stored
        :tpr_threshold: float, should be a value in the unit interval, used for gathering the performance of the oodd tests
        """
        self.name = params['name'] if 'name' in params.keys() else None
        self.device = params['device'] if 'device' in params.keys() else torch.device('cpu')
        self.model = params['model']
        self.model_address = params['model_address'] if 'model_address' in params.keys() else os.path.join(os.getcwd(),'model_address')
        self.data_address = params['data_address'] if 'data_address' in params.keys() else os.path.join(os.getcwd(),'data_address')
        self.model_uq_address =  params['model_uq_address'] if 'model_uq_address' in params.keys() else self._make_model_uq_address() 
        self.tpr_threshold =  params['tpr_threshold'] if 'tpr_threshold' in params.keys() else 0.95
        self.n_class = params['n_class']
        self.oodd_tests = dict()
        #self.oodd_test_results = dict()

    def __str__(self):
        return f"{self.name}\n(model_address {self.model_address})\n(data address {self.data_address})\n(model_uq address {self.model_uq_address})"
        
    def _make_model_uq_address(self):
        """ Internal, effectively private function, for making the model_uq folder if not presently found in specified path        
        """
        if not self.name:
            MODEL_NAME_UQ = "Recent_UQ"
        else:
            MODEL_NAME_UQ = self.name +"_UQ"
        if not os.path.exists(os.path.join(os.getcwd(),MODEL_NAME_UQ)):
            os.makedirs(os.path.join(os.getcwd(),MODEL_NAME_UQ))
        self.model_uq_address = os.path.join(os.getcwd(),MODEL_NAME_UQ)

    def to_dict(self):
        """ Convert instance attributes to a dictionary."""
        return self.__dict__
        
    def to_json(self):
        """ Serialize attributes to a JSON string."""
        return json.dumps(self.to_dict())
        
    def save(self, address:str, name:Optional[str]=None):
        """ Save a pickled or else JSON string representing the attribute of the model uq object"""
        self.model_uq_address = address
        dict_to_save = self.to_dict()
        if name is None:
            filename = '_model_uq.pickle'
        else:
            filename = name+'.pickle'
        with open(os.path.join(address,filename),'wb') as handle:
            pickle.dump(dict_to_save, handle,protocol=pickle.HIGHEST_PROTOCOL)
        print('Model saved at {}'.format(address))

    def make_certainties(self,
                         model_data_address:Optional[str]=None,
                         model_data_name:Optional[str]=None,
                         model_data: Optional[Model_Data]=None,
                         certainties_name:Optional[str]=None,
                         is_bayesian=False,
                         select_rate=.9,
                         select_sum=20,
                         return_certainties=True):
        """ Method for making the certainty data structure and then running the gather certainties therein.
        """
        # load in model_data_address
        if not model_data:
            model_data = Model_Data(model_data_address=model_data_address,model_data_filename=model_data_name)
        # check if model certainties are saved
        if not certainties_name:
            certainties_name = '_certainties'
        if not os.path.exists(self.model_uq_address):
            os.makedirs(self.model_uq_address)
        if not os.path.exists(os.path.join(self.model_uq_address,certainties_name)):
            certainties = Certainties()
            certainties.gather_certainties(output_data=model_data,
                                           is_bayesian=is_bayesian,
                                           select_rate=select_rate,
                                           select_sum=select_sum)
            certainties.save(address=self.model_uq_address,name=certainties_name)
            del model_data
        if return_certainties:
            gc.collect()
            return certainties
        else:
            del certainties
            gc.collect()
            
    def form_model_data(self,
                        data:dict[str,Union[torch.Tensor,np.array,str,bool]],
                        apply_softmax=True,
                        custom_weight_function=None,
                        is_bayesian=False,
                        **additional_params):
        """ Generate Model_Data object from data dictionary.

        Parameters
        ---------------
        :data: dict, described in the Model_Data class
        :apply_softmax: bool, default=True, whether a softmax layer will be applied
        :custom_weight_function: optional function to post-compose with the model to weight,ensemble outputs
        :is_bayesian: bool, default=False, indicates that the data is drawn from a Bayesian sample, may require a custom
        weight function.
        """
        return Model_Data(model = self.model,
                          data = data,
                          apply_softmax = apply_softmax,
                          custom_weight_function = custom_weight_function,
                          is_bayesian = is_bayesian,
                          **additional_params)

    def gen_model_data(self,
                       data:Optional[dict[str,Union[torch.Tensor,np.array,str,bool]]] = None,
                       apply_softmax = True,
                       custom_weight_function = None,
                       is_bayesian = False,
                       **additional_params):
        """ Check if model_data object is saved, and if filepath found, will load in the model_data. Otherwise, will generate model_data from inputs.
        """
        try:
            if not os.path.exists(self.data_address):
                os.makedirs(self.data_address)
            if not os.path.exists(os.path.join(self.data_address,'_model_data.pickle')):
                model_data = self.form_model_data(data = data,
                                                  apply_softmax = apply_softmax,
                                                  custom_weight_function = custom_weight_function,
                                                  is_bayesian = is_bayesian,
                                                  **additional_params)
                model_data.save(address=self.data_address,name='_model_data')
                return model_data
            else:
                model_data = Model_Data()
                model_data.load(address=self.data_address,name='_model_data')
                return model_data
        except Exception as E:
            print("While gathering model data, ran into the following error:{}".format(E))

    def make_assignment_df(self,
                           assignment_df_address:Optional[str]=None,
                           assignment_df_name:Optional[str]=None,
                           prediction:Optional[Union[pd.Series,np.ndarray,torch.Tensor]]=None,
                           classification:Optional[Union[pd.Series,np.ndarray,torch.Tensor]]=None,
                           predictions:Optional[Union[pd.Series,np.ndarray,torch.Tensor]]=None,
                           predictive_status:Optional[Union[pd.Series,list[str]]]=None,
                           assignment_df:Optional[pd.DataFrame]=None,
                           return_assignment_df=True):
        """ Generate assignment_df object from provided inputs.

        Parameters
        ------------
        :assignment_df_address: folder under which the assignment df may be initialized from, if available, and to be saved in otherwise
        :assignment_df_name: name under which assignment_df may be found from path if available, and to be saved as otherwise
        :prediction: 1-d Series, ndarray, or Tensor object, displaying in an indexed fashion the model prediction 
        :classification: 1-d Series, ndarray, or Tensor object, displaying in an index fashion the ground truth for the indexed data, if available
        :predictions: Series, ndarray, or Tensor object, displaying in an indexed fashion additional predictions (e.g. top k, etc)
        :predictive_status: indexed display identifying if the predicted values are True positives, False positives, or unknown
        :assignment_df: Optional dataframe object to pass and store in a Assignment_DF class instance.
        :return_assignment_df: bool to indicate if the Assignment DF object is to store in memory, or otherwise be removed from memory once being saved and stored on disk 
        """
        assignment_df = Assignment_DF(assignment_df_address=assignment_df_address,
                                      assignment_df_name=assignment_df_name,
                                      prediction=prediction,
                                      classification=classification,
                                      predictions=predictions,
                                      predictive_status=predictive_status,
                                      assignment_df=assignment_df)
        assignment_df.save(address=self.data_address,name=assignment_df_name)
        print("Assignment DF saved at {}".format(self.data_address))
        if return_assignment_df:
            return assignment_df
        else:
            del assignment_df
        
                           
    def make_certainty_dist(self,
                            model_data_name:Optional[str]=None,
                            model_data:Optional[Model_Data]=None,
                            certainties_name:Optional[str]=None,
                            certainties:Optional[Certainties]=None,
                            is_bayesian=False,
                            return_certainty_dist = False):
        """ Method for making the Certainty Distribution object from model data objects and certainty objects from the Model Data and Certainties objet

        Parameters:
        --------------------
        :model_data_name: name of the Model Data to load in and form the distribution from, with the model_data_address saved as a model_uq attribute
        :model_data: optional Model Data object to load in and form the distribution from
        :certainties_name: name of the Certainties object to load in and form the distribution from, with the certainties path address defaulting to the model_uq_address attribute
        :certainties: optional Certainties object to directly form the distribution from
        :is_bayesian: bool, default=False, indicates the certainty distribution should be formed from the ensembled data
        :return_certainty_dist: bool, default=False, return the certainty distribution class and keep in memory or delete
        """
        try:
            if self.name:
                cert_name = self.name+"_certainty_dist"
            else:
                cert_name = "_certainty_dist"
            if not os.path.exists(self.model_uq_address):
                os.makedirs(self.model_uq_address)
            if not os.path.exists(os.path.join(self.model_uq_address,cert_name)):
                if is_bayesian:
                    _args = dict({'classification':model_data.input_data_labeled,
                                      'cat_predict':certainties.ensemble_predictions,
                                      'cert_score':certainties.ensemble_certainty_score,
                                      'predictions':model_data.ensemble_output,
                                      'certainty':certainties.ensemble_certainty,
                                      'is_bayesian':is_bayesian,
                                     })
                else:
                    _args = dict({'classification':model_data.input_data_labeled,
                                  'cat_predict':certainties.predictions,
                                  'cert_score':certainties.certainty_score,
                                  'predictions':model_data.output,
                                  'certainty':certainties.certainty,
                                  'is_bayesian':is_bayesian,
                                 })
                certainty_dist = Certainty_Distribution(find_certainty_dist_dataframe(test_data=False,**_args))
                certainty_dist.save(address=self.model_uq_address, name = cert_name)
                if return_certainty_dist:
                    return certainty_dist
                else:
                    del certainty_dist
                    gc.collect()
            else:
                if return_certainty_dist:
                    return Certainty_Distribution(address = os.model_uq_address,name=cert_name)
        except Exception as E:
            print(E)    
            
    def gather_omicron_data(self,
                            internal_omicrons:bool,
                            assignment_df_name:Optional[str]=None,
                            assignment_df:Optional[Assignment_DF]=None,
                            scores_name:Optional[str]=None,
                            scores:Optional[Scores]=None,
                            certainty_dist_name: Optional[str]=None,
                            certainty_dist:Optional[Certainty_Distribution]=None,
                            certainties_name:Optional[str]=None,
                            certainties:Optional[Certainties]=None,
                            omicrons_name:Optional[str]=None,
                            tpr_threshold=.95,
                            is_bayesian=False,
                            return_omicrons=False,
                            omicron_test_params : Optional[dict]=None):
        """ Form the omicron data from the corresponding Assignment DF, Scores, Certainty Dist, and Certainties objects

        Parameters
        -----------------------------------
        :internal_omicrons: bool, indicating if the omicrons are being computed with respect to internal data, when called as an internal method, will default to True
        :assignment_df_name: optional string name indicating the assignment_df to load in
        :assignment_df: optional Assignment DF object to pass through if already loaded in memory
        :scores_name: optional string indicating the scores object to load in
        :scores: optional Scores object to pass through if already loaded in memory
        :certainty_dist_name: optional string indicating the name of the Certainty Distribution object to load in
        :certainty_dist: optional Certainty_Distribution object to pass through
        :certainties_name: optional string indicating the name of the Certainties object to load in
        """
        omicrons = Omicron_Data()
        omicrons.gather_omicrons(assignment_df_address=self.data_address,
                                 certainty_dist_address=self.model_uq_address,
                                 certainties_address=self.model_uq_address,
                                 scores_address=self.model_uq_address,
                                 assignment_df_name=assignment_df_name,
                                 certainty_dist_name=certainty_dist_name,
                                 scores_name=scores_name,
                                 certainties=certainties,
                                 certainty_dist=certainty_dist,
                                 assignment_df = assignment_df,
                                 scores=scores,
                                 is_bayesian=is_bayesian)
        self.oodd_tests.__setitem__('Omicron Test',omicrons.make_omicron_test(n_class=self.n_class,
                                                                              tpr_threshold=tpr_threshold))
        omicrons.save(address=self.model_uq_address,name=omicrons_name)
        if return_omicrons:
            return omicrons
        else:
            del omicrons
                            
    def fill_uq(self,
                data:Optional[dict[str,Union[torch.Tensor,np.array,str,bool]]] = None,
                model_data:Optional[Model_Data] = None,
                assignment_df_name:Optional[str] = None,
                model_data_name:Optional[str] = None,
                certainty_dist_name:Optional[str] = None,
                certainties_name:Optional[str] = None,
                scores_name:Optional[str] = None,
                omicrons_name:Optional[str] = None,
                apply_softmax = True,
                custom_weight_function = None,
                is_bayesian = False,
                select_rate = .9,
                select_sum = 20,
                return_model_data = False,
                return_assignment_df = False,
                return_certainties =False,
                return_certainty_dist = False,
                return_scores = False,
                return_omicrons = False,
                verbose=False,
                tpr_threshold=.95,
                omicron_test_params : Optional[dict]=None,
                **additional_params)-> Optional[dict[str,Union[Model_Data,Assignment_DF,Certainties,Certainty_Distribution,Scores,Omicron_Data]]]:
        """
        One stop method for filling in model_data, certainties, certainty_distribution, and scores data in separate files/objects. 
        
        Parameters
        -----------
        :data:Optional[dict[str,Union[torch.Tensor,np.array,str,bool]]]=None, data to seed model_data object generation if it does not already
        exist
        :assignment_df_name:Optional[str]=None, optional name for assignment_df object to load
        :model_data_name:Optional[str]=None, optional name for model_data
        :certainty_dist_name:Optional[str]=None, optional name for certainty distribution data
        :certainties_name:Optional[str]=None, optional name for certainties data
        :scores_name:Optional[str]=None, optionalname for scores data
        :omicrons_name: Optional[str]=None, optional name for the omicrons data (recommended if preparing multiple samples;batches)
        :apply_softmax: bool, determines if softmax adjustment needs to be post-composed with model output
        :custom_weight_function: alternate function to reweight ensemble models
        :is_bayesian: bool, determines if alternate applications need to be run to account for ensembling Bayesian samples
        :select_rate: indicates the final portion of Bayesian samples to ensemble, i.e. starting at the 90% of samples
        :select_sum: indicates the final number of Bayesian samples to ensemble
        :return_model_data: bool, indicates that the model_data should be held in memory after use
        :return_assignment_df: bool, indicates the assignment_df should be held in memory after generation
        :return_certainties: bool, indicates that should the certainties be generated, they're held in memory
        :return_certainty_dist: bool, indicates that should the certainty_dist need to be generated, it is held in memory
        :return_scores: bool, indicates that scores should be held in memory 
        :return_omicrons: bool, indicates that omicrons should be held in memory once generated
        :additional_params: dict of additional parameters to be fed into hamiltorch, or other Bayesian samplers
        """
        # We want to hold onto the model data to generate the assignment_df
        if is_bayesian:
            prediction_type = 'ensemble_prediction'
            predictions_type = 'ensemble_predictions'
        else:
            prediction_type = 'prediction'
            predictions_type = 'predictions'
        if not model_data:
            model_data = self.gen_model_data(data = data,
                                             apply_softmax = apply_softmax,
                                             custom_weight_function = custom_weight_function,
                                             is_bayesian = is_bayesian,
                                             **additional_params)
        if verbose:
            print(model_data.__dict__)
        gatherlist1 = model_data.make_full_gatherlist()
        gatherlist2 = ['Global']+list(model_data.classification_categories.keys())
        safevalues = model_data['safevalues'] if 'safevalues' in model_data.to_dict().keys() else None
        classification_categories = model_data.classification_categories
        if verbose:
            print("Full gatherlist :{}".format(gatherlist1))
            print("Complementary gatherlist: {}".format(gatherlist2))
            print("Given safevalues (if any): {}".format(safevalues))
            print("Classification categories: {}".format(classification_categories))
        # Make Assignment DF
        try:
            if return_assignment_df:
                assignment_df= self.make_assignment_df(assignment_df_address=self.data_address,                            
                                                       assignment_df_name=assignment_df_name,                              
                                                       prediction=getattr(model_data,prediction_type,None),
                                                       classification=getattr(model_data,'input_data_labeled',None),  
                                                       predictions=getattr(model_data,predictions_type,None),
                                                       predictive_status=None,                            
                                                       assignment_df=None,                            
                                                       return_assignment_df=return_assignment_df)
                if verbose:
                    print("The assignment dataframe object is: {}".format(assignment_df.to_dict()))

            else:
                self.make_assignment_df(assignment_df_address=self.data_address,
                                        assignment_df_name=assignment_df_name,
                                        prediction =getattr(model_data,prediction_type,None),
                                        classification=getattr(model_data,'input_data_labeled',None),
                                        predictions =getattr(model_data,predictions_type,None),   
                                        predictive_status=None,                                                            
                                        assignment_df=None,                            
                                        return_assignment_df=return_assignment_df)
                assignment_df=None
            if not return_model_data:
                del model_data
                gc.collect()
                model_data = None
        # Make the Certainties Data
            if return_certainties:
                certainties = self.make_certainties(model_data_address = self.data_address,                    
                                                    model_data_name = model_data_name,                             
                                                    certainties_name = certainties_name,                          
                                                    is_bayesian = is_bayesian,                        
                                                    select_rate = select_rate,                          
                                                    select_sum = select_rate,                            
                                                    return_certainties = return_certainties)
                if verbose:
                    print("The certainties objects is: {}".format(certainties.__dict__))
            else:
                certainties = None
                self.make_certainties(model_data_address = self.data_address,   
                                      model_data_name = model_data_name, 
                                      certainties_name = certainties_name,
                                      is_bayesian = is_bayesian, 
                                      select_rate = select_rate, 
                                      select_sum = select_rate, 
                                      return_certainties = return_certainties)
            # Make the certainty_dist_data
            if return_certainty_dist:
                certainty_dist = self.make_certainty_dist(model_data_name = model_data_name,
                                                      model_data = model_data,
                                                      certainties_name = certainties_name,
                                                      certainties = certainties,
                                                      is_bayesian = False,
                                                      return_certainty_dist = return_certainty_dist)
                if verbose:
                    print("The certainty distribution {}".format(certainty_dist.__dict__))
            else:
                self.make_certainty_dist(model_data_name=model_data_name,                                           
                                    model_data=model_data,                                
                                    certainties_name = certainties_name,                               
                                    certainties = certainties,                               
                                    is_bayesian=False,                              
                                    return_certainty_dist = return_certainty_dist)
                certainty_dist = None
        # make Scores
            print("Generating scores object")
            scores = Scores(tpr_threshold=self.tpr_threshold)
            # run equivalent of self._apply_classification_scheme()
            if safevalues:
                scores.get_bin_scores(assignment_df_address= self.data_address,
                                      assignment_df_name = assignment_df_name,
                                      model_data_address = self.data_address,
                                      model_data_name = model_data_name,
                                      certainties_address = self.model_uq_address,
                                      certainties_name = certainties_name,
                                      is_bayesian=is_bayesian)
            else:
                scores.get_scores(assignment_df_address= self.data_address,
                                  assignment_df_name = assignment_df_name,
                                  model_data_address = self.data_address,
                                  model_data_name = model_data_name,
                                  certainties_address = self.model_uq_address,
                                  certainties_name = certainties_name,
                                  is_bayesian=is_bayesian)                
            scores.gather_certainty_score_stats(total_gatherlist=gatherlist1,
                                                predictive_comparison_gatherlist=gatherlist2,
                                                assignment_df_address=self.data_address,
                                                assignment_df_name=assignment_df_name,
                                                certainties_address = self.model_uq_address,
                                                certainties_name = certainties_name,
                                                certainties = certainties,
                                                is_bayesian=is_bayesian)
            scores.get_empirical_competencies(certainty_dist_address=self.model_uq_address,
                                              certainty_dist_name=certainty_dist_name,
                                              certainty_dist=certainty_dist,
                                              classification_categories=classification_categories)
            scores.get_component_competencies(model_data_address=self.model_uq_address,
                                              model_data_name=model_data_name,
                                              model_data=model_data,
                                              classification_categories=classification_categories,
                                              is_bayesian=is_bayesian)
            scores.save(address=self.model_uq_address, name=scores_name)
            if verbose:
                print("The scores are {}".format(scores.__dict__))
            if return_omicrons:
                omicrons =self.gather_omicron_data(internal_omicrons=True,
                                                   assignment_df_name=assignment_df_name,
                                                   assignment_df=assignment_df,
                                                   tpr_threshold=tpr_threshold,
                                                   is_bayesian=is_bayesian,
                                                   scores_name=scores_name,
                                                   scores=scores,
                                                   certainty_dist_name=certainty_dist_name,
                                                   certainty_dist=certainty_dist, 
                                                   certainties_name=certainties_name,     
                                                   certainties=certainties,
                                                   omicrons_name=omicrons_name,
                                                   return_omicrons=return_omicrons,
                                                   omicron_test_params =omicron_test_params)
                if verbose:
                    print("The omicrons are {}".format(omicrons.__dict__))
            else:
                self.gather_omicron_data(internal_omicrons=True,
                                         assignment_df_name=assignment_df_name,
                                         assignment_df=assignment_df,
                                         tpr_threshold=tpr_threshold,
                                         is_bayesian=is_bayesian,
                                         scores_name=scores_name,
                                         scores=scores,
                                         certainty_dist_name=certainty_dist_name,
                                         certainty_dist=certainty_dist, 
                                         certainties_name=certainties_name,  
                                         certainties=certainties,
                                         omicrons_name=omicrons_name,
                                         return_omicrons=return_omicrons,
                                        omicron_test_params=omicron_test_params)
            if is_bayesian:
                self.oodd_tests.__setitem__('PCS-VRO Test',scores.gather_pcs_vro_stats(classification_gatherlist=gatherlist1,
                                                                                      certainties_address=self.model_uq_address,
                                                                                      certainties_name=certainties_name,
                                                                                      certainties = certainties,
                                                                                      return_pcs_vro_test=True))
            if not return_scores:
                del scores        
                gc.collect()
            if self.name:
                model_uq_name = self.name+"_model_uq"
            else:
                model_uq_name = 'model_uq'
            self.save(address=self.model_uq_address,name=model_uq_name)
            return_dict=dict()
            if return_model_data:
                return_dict.__setitem__('model_data',model_data)
            if return_assignment_df:
                return_dict.__setitem__('assignment_df',assignment_df)
            if return_certainties:
                return_dict.__setitem__('certainties',certainties)
            if return_certainty_dist:
                return_dict.__setitem__('certainty_dist',certainty_dist)
            if return_scores:
                return_dict.__setitem__('scores',scores)
            if return_omicrons:
                return_dict.__setitem__('omicrons',omicrons)
            if len(list(return_dict.keys()))>0:
                return return_dict
        except Exception as E:
            print("While filling in model uncertainty quantification data, we raised the following error:{}".format(E))

    def run_oodd_tests_internal(self,
                                external_data_name:str,
                                internal_certainties_name:Optional[str]=None,
                                internal_certainties:Optional[Certainties]=None,
                                internal_omicrons_name:Optional[str]=None,
                                internal_omicrons:Optional[Omicron_Data]=None,
                                external_data_address:Optional[str]=None,
                                external_data:Optional[Certainties]=None,
                                scores_address:Optional[str]=None,
                                scores_name:Optional[str]=None,
                                scores:Optional[Scores]=None,
                                omicron_data_name: Optional[str]=None,
                                is_bayesian=False,
                                external_is_bayesian=False,
                                omicron_test_params : Optional[dict]=None):
        """ Command that applies the internally derived out-of-distribution detection tests stored within the model_uq 
        class to external data provided in the form of a Certainties class object. The current fixed test is the 
        omicron detection test. However, if the data is indicated to be bayesian, we also attempt to apply the PCS-VRO test.
        :external_data_name:str, determines the saved file name for the external Certainty object for loading and saving results
        :internal_certainties_name:Optional[str]=None,  string name for loading in internal certainty data
        :internal_certainties:Optional[Certainties]=None, alternative to use if internal certainties have already been loaded into memory
        :internal_omicrons_name:Optional[str]=None,  string for identifying and loading in the internal omicron data
        :internal_omicrons: Optional[Omicron_Data]=None, alternative to use if the internal omicrons have already been loaded into memory
        :external_data_address:Optional[str]=None, string identifying the folder where the external certainty object is store
        :external_data:Optional[Certainties]=None,
        :scores_address:Optional[str]=None,
        :scores_name:Optional[str]=None,
        :scores:Optional[Scores]=None, 
        :omicron_data_name: Optional[str]=None,
        :is_bayesian:bool, Default = False,
        :external_is_bayesian: bool, Default =False,
        
        """
        novel_stats_dict = dict()
        print('Loading in the external data')
        if not internal_certainties:
            try:
                internal_certainties = Certainties(address = self.model_uq_address, name = internal_certainties_name)
            except Exception as E:
                print("While loading in the internal data, we raised the following error:\n{}".format(E))
        if not internal_omicrons:
            try:
                internal_omicrons = Omicron_Data()
            except Exception as E:
                print("While loading in the internal omicrons, we raised the following error:\n{}".format(E))
        if not external_data:
            try:
                external_data = Certainties(address=external_data_address, name=external_data_name)
            except Exception as E:
                print("While loading in the external data, we raised the following error:\n {}".format(E))
        print('Starting Omicron tests')
        ex_output = external_data.ensemble_output if external_is_bayesian else external_data.output
        testparams = dict({'n_class' : self.n_class,
                           'predictive_status' : 'Global',
                           'tpr_threshold' : self.tpr_threshold})
        if omicron_test_params is not None:
            testparams.update(testparams)
        test_dict = omicron_test_results(test_sample = ex_output,
                                         omicron_d = internal_omicrons.omicrons,
                                         internal_test = self.oodd_tests['Omicron Test'],
                                         **testparams)
        novel_stats_dict.__setitem__('Omicron Test Results',test_dict)
        if is_bayesian:
            print('Starting PCS VRO Test')
            gatherlist=product(['Global'],[''])
            if 'PCS-VRO Test' not in self.oodd_tests.keys():
                print("PCS-VRO test not found. Proceeding to make")
                gatherlist = product(['Global'],[''])
                self.oodd_tests.__setitem__('PCS-VRO Test',scores.gather_pcs_vro_stats(classification_gatherlist = gatherlist,
                                                                                       certainties_address = self.model_uq_address,
                                                                                       certainties_name = internal_certainties_name,
                                                                                       certainties = internal_certainties,
                                                                                       return_pcs_vro_test = True)
                                           )
            if 'Global' not in self.oodd_tests['PCS-VRO Test'].keys():
                print("PCS-VRO test keys do not include global information. Proceeding to make")
                gatherlist=product(['Global'],[''])
                self.oodd_tests.__setitem__('PCS-VRO Test',scores.gather_pcs_vro_stats(classification_gatherlist = gatherlist,
                                                                                       certainties_address = self.model_uq_address,
                                                                                       certainties_name = internal_certainties_name,
                                                                                       certainties = internal_certainties,
                                                                                       return_pcs_vro_test = True)
                                           )
            out_pcs = external_data.ensemble_certainty_score.to(self.device)
            out_vro  = vro(external_data.ensemble_predictions,external_data.predictions).to(self.device)
            in_pcs = internal_certainties.ensemble_certainty_score.to(self.device)
            in_vro = internal_certainties.VRO.to(self.device)
            print("THE PCS-VRO TEST KEYS ARE: {}".format(self.oodd_tests['PCS-VRO Test'].keys()))
            #pcs_vro_test = self.oodd_tests['PCS-VRO Test']['Global']['PCS-VRO Test']
            HP = self.oodd_tests['PCS-VRO Test']['Global']['HPCS'].to(self.device)
            LP = self.oodd_tests['PCS-VRO Test']['Global']['LPCS'].to(self.device)
            HV = self.oodd_tests['PCS-VRO Test']['Global']['HVRO'].to(self.device)
            LV = self.oodd_tests['PCS-VRO Test']['Global']['LVRO'].to(self.device)

            # If the size of the in_pcs and in_vro are 'reasonable, then we can run the pcs_vro tests; if they're too large, we
            # need to make a temporary dataloader, and aggregate the scores
            in_score = pcs_vro_test(pcs=in_pcs,vro=in_vro,HPCS=HP,LPCS=LP,HVRO=HV,LVRO=LV)
            out_score = pcs_vro_test(pcs=out_pcs,vro=out_vro,HPCS=HP,LPCS=LP,HVRO=HV,LVRO=LV)
            out_summary = pcs_vro_test_summary(out_pcs,out_vro,HPCS=HP,LPCS=LP,HVRO=HV,LVRO=LV)
            scores= torch.cat([in_score,out_score])
            labels = torch.cat([torch.zeros(in_score.shape[0]),torch.ones(out_score.shape[0])])
            metrics = oodd_metrics_internal(scores=scores,
                                            labels=labels,
                                            tpr_threshold=self.tpr_threshold)
            materials = dict({"in_score":in_score.cpu(),
                              "out_score":out_score.cpu(),
                              "thresholds":dict({"HPCS":HP.cpu(),
                                                 "LPCS":LP.cpu(),
                                                 "HVRO":HV.cpu(),
                                                 "LVRO":LV.cpu(),
                                                }),
                              "summary":out_summary,})
            novel_stats_dict.__setitem__("PCS-VRO Test",metrics)
            novel_stats_dict.__setitem__("PCS-VRO Test Materials",materials)
        internal_test_name = external_data_name +"_oodd_test_results"
        adr = os.path.join(self.model_uq_address, internal_test_name)
        with open(adr,'wb') as handle:
            pickle.dump(novel_stats_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.save(self.model_uq_address,name=self.name)    
                
    def run_oodd_test_external(self,
                               detector,
                               combined_dataloader:DataLoader,
                               external_data_name:str,
                               detector_name:str,
                               external_save_address:str,
                               device='cpu'):
        """ Method for applying an external out-of-distribution detectors from the pytorch_ood repository to a pre-formed DataLoader that combines
        labeled in and out of distribution data
        """
        novel_stats_dict=dict()
        print("Running {} Detection Test on device {}".format(detector_name,device))
        novel_stats_dict.update(oodd_test_and_metrics_external(detector = detector,
                                                               detector_name = detector_name,
                                                               combined_dataloader = combined_dataloader,
                                                               device = device,
                                                               tpr_threshold = self.tpr_threshold)
                               )
        del detector
        torch.cuda.empty_cache()
        gc.collect()
        test_name = external_data_name+"_"+ detector_name +"_oodd_test_results"
        adr = os.path.join(external_save_address, internal_test_name)
        with open(adr,'wb') as handle:
            pickle.dump(novel_stats_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.save(address= self.model_uq_address,name = self.name)

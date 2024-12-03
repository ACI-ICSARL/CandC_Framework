# omicron_data.py
import torch
import pandas as pd
import os
import pickle

from typing import Union, Optional
from .gather import *

from .assignmentdf import Assignment_DF
from .certainties import Certainties
from .certainty_distribution import Certainty_Distribution
from .model_data import Output_Data,Model_Data
from .scores import Scores

from ...oodd.omicrons import *

class Omicron_Data():
    f""" Class object intended to transform certainties info and output_data/model_data
    attributes into a serialized structure organized by label/category.

    """
    def __init__(self): 
            self.omicrons=dict()
    
    def gather_omicrons(self,
                        assignment_df_address:str,                   
                        certainty_dist_address:str,                       
                        certainties_address:str,                       
                        scores_address:str,                        
                        assignment_df_name:Optional[str]=None,   
                        assignment_df:Optional[Assignment_DF]=None,
                        certainty_dist_name:Optional[str]=None,    
                        certainty_dist:Optional[Certainty_Distribution]=None,
                        certainties_name:Optional[str]=None,                 
                        certainties:Optional[Certainties]=None,
                        scores_name:Optional[str]=None,
                        scores: Optional[Scores]=None,
                        is_bayesian=False):
        """
        Our goal is to fill the omicrons subobject and produce an omicron test for the oodd_test subobject.
        Parameters:
        ----------------------
        :assignment_df_address: folder address containing the Assignment DF object to load
        :certainty_dist_address: folder address containing the Certainty Distribution object to load
        :certainties_address: folder address containing the Certainties object to load
        :scores_address: folder address containing the Scores object to load
        :is_bayesian: bool, indicates if the data is to be processed as a Bayesian sample

        Optional Parameters:
        -------------------
        :assignment_df_name: optional name identifying the Assignment DF object to load
        :certainty_dist_name: optional name identifying the Certainty Distribution object to load
        :certainties_name: optional name identifying the Certainties object to load
        :scores_name: optional name identifying the Scores object to load
        """
        if is_bayesian:
            ref='ensemble_certainty'
        else:
            ref='certainty'
        # load in assignment_df from address
        try:#try:
            if assignment_df is None:
                assignment_df = Assignment_DF(assignment_df_address = assignment_df_address,
                                              assignment_df_name = assignment_df_name)
        # load in certainty_dist from address
            if certainty_dist is None:
                certainty_dist = Certainty_Distribution(address = certainty_dist_address,
                                                         name = certainty_dist_name)
        # load in certainties from address
            if certainties is None:
                certainties = Certainties(address = certainties_address, 
                                          name = certainties_name)
        # load in scores
            if scores is None:
                scores = Scores(address = scores_address, 
                                name = scores_name)
        #Make gatherlist
        except Exception as E:
            print("While loading in necessary structures, we raised the following error: {}".format(E))
        gatherlist = list(assignment_df.data['classification'].unique())
        print("The omicron gatherlist has {} many categories".format(len(gatherlist)))
        print("Gathering the omicrons for all observed labels")
        for cat in tqdm(gatherlist,desc="Gathering omicron data"):
            try: #try:
                self._gather_cat_omicrons(cat = cat,
                                         assignment_df = assignment_df,
                                         certainty_dist = certainty_dist,
                                         certainties = certainties,
                                         scores = scores,
                                         is_bayesian = is_bayesian
                                        )
            except Exception as E:
                print("The following error was raised while trying to gather internal omicrons\n {}".format(E))
                
    def _gather_cat_omicrons(self,
                             cat:Union[str,int],
                             assignment_df:Assignment_DF,
                             certainty_dist:Certainty_Distribution,
                             certainties:Certainties,
                             scores:Scores,
                             is_bayesian=False):
        """
        Compute the omicrons from the self.data.y_test data by predicted category.
        Parameters
        --------------------
        :cat: Integer or String value indicating selected category/label from which we compute the
        omicron scores.
        :assignment_df: Assignment_DF class object containing the assignment pandas dataframe
        :certainty_dist: Certainty_Distribution class object containing the certainty_distribution 
        pandas dataframe
        :certainties: Certainties class object containing the relevant certainties 
        :scores: Scores class object containing the empirical competencies
        :is_bayesian: boolean, used to indicate if the data is a Bayesian sample
        """
        sub_object=dict()
        # GLOBAL
        sub_object.__setitem__('Global',self.gather_internal_omicron_data(cat=cat,
                                                                          predictive_status=None,
                                                                          assignment_df=assignment_df.data,
                                                                          certainty_dist=certainty_dist.data,
                                                                          certainties=certainties,
                                                                          is_bayesian=is_bayesian))
        # TP
        sub_object.__setitem__('TP',self.gather_internal_omicron_data(cat=cat,
                                                                      predictive_status='TP',
                                                                      assignment_df=assignment_df.data,
                                                                      certainty_dist=certainty_dist.data,
                                                                      certainties=certainties,
                                                                      is_bayesian=is_bayesian))
        # FP
        sub_object.__setitem__('FP',self.gather_internal_omicron_data(cat=cat,
                                                                      predictive_status='FP',
                                                                      assignment_df=assignment_df.data,
                                                                      certainty_dist=certainty_dist.data,
                                                                      certainties=certainties,
                                                                      is_bayesian=is_bayesian))
        # Set the empirical competence within category
        sub_object.__setitem__('empirical_competence',getattr(scores,'empirical_competencies',dict({}) ).get(cat,None))
        # attach the subobject dict to the category for the omicrons
        self.omicrons.__setitem__(cat,sub_object) 
    
    def gather_internal_omicron_data(self,
                                     cat:Union[str,int],
                                     assignment_df:pd.DataFrame,
                                     certainty_dist:pd.DataFrame,
                                     certainties:Certainties,
                                     predictive_status=None,
                                     is_bayesian = False
                                    ):
        """ This function returns the subobject dictionary corresponding to the category entry of the internal
        omicron object. It consists of the corresponding omicrons by category and predictive status, a vector
        of the predictive statuses of the of the correspsonding omicrons, and finally a sample of the 
        corresponding certainties for rapid calculation of future omicrons.
        Parameters
        --------------------
        :cat: Integer or String value indicating selected category/label from which we compute the
        omicron scores.
        :assignment_df: Assignment_DF class object containing the assignment pandas dataframe
        :certainty_dist: Certainty_Distribution class object containing the certainty_distribution 
        pandas dataframe
        :certainties: Certainties class object containing the relevant certainties 
        :predictive_status: None, Global, 'TP', 'FP' used to identify the subset of data analyzed
        """
        omicrons = self._make_omicrons(cat = cat,
                                       predictive_status = predictive_status,
                                       assignment_df = assignment_df,
                                       certainties = certainties,
                                       is_bayesian = is_bayesian)
        dim = omicrons.shape[0]
        ps = self._make_anomalous_predictive_status(cat = cat,
                                                    dim = dim,
                                                    predictive_status = predictive_status,
                                                    certainty_dist = certainty_dist)
        certainty_sample=self._get_certainty_sample(cat = cat,
                                                    certainty_dist = certainty_dist,
                                                    omicrons = omicrons,
                                                    certainties = certainties,
                                                    predictive_status = predictive_status,
                                                    sample_size_max = 50,
                                                    is_bayesian=is_bayesian)
        output = dict({'omicrons': omicrons,
                       'predictive_statuses': ps,
                       'certainty_sample': certainty_sample})
        return output
        
    def _make_omicrons(self,
                       cat:Union[int,str],
                       assignment_df:pd.DataFrame,
                       certainties:Certainties,
                       predictive_status:Optional[str]=None,
                       is_bayesian=False):
        """ Given a category, we wish to produce the in_sample_indices corresponding to relevant certainties,
        and then use these to produce the omicrons
        Parameters
        --------------------
        :cat: Integer or String value indicating selected category/label from which we compute the
        omicron scores.
        :assignment_df: Assignment_DF class object containing the assignment pandas dataframe
        :certainties: Certainties class object containing the relevant certainties 
        :is_bayesian: boolean, used to indicate if the data is a Bayesian sample
        :predictive_status: None, Global, 'TP', 'FP' used to identify the subset of data analyzed
        """
        tempdf= assignment_df.reset_index()        
        if predictive_status is not None:
            in_sample_indices = tempdf.loc[(tempdf.prediction==cat) &(tempdf.predictive_status==predictive_status)].index.tolist()
        else:
            in_sample_indices = tempdf.loc[tempdf.prediction==cat].index.tolist()
        if is_bayesian:
            X = certainties.ensemble_certainty[in_sample_indices]
        else:
            X = certainties.certainty[in_sample_indices]
        return omicron_fn(X,X)
        
    def _make_anomalous_predictive_status(self,
                                          dim:int,
                                          cat:Union[int,str],
                                          certainty_dist:pd.DataFrame,
                                          predictive_status:Optional[str]=None
                                         ):
        """
        We wish to output a vector of 0/1 values. Run to ensure that the predictive statuses for omicron test
        are properly calibrated. If predictive status is TP, we set to 0. If predictive status is FP, 
        we set to 1.

        Parameters
        ----------
        :dim: int, indicating the number of samples
        :cat: Union[int,str]. By default, should be an integer, indicating the category selected for
        :predictive_status: str. Should be one of two values 'TP' or 'FP', or otherwise ''.
        :certainty_dist: a pd.DataFrame derived from the Certainty_Distribution class.
        
        """
        try:
            if predictive_status=='TP':
                ps = torch.ones(dim)
            elif predictive_status=='FP':
                ps = torch.zeros(dim)
            else:
                tempps = certainty_dist.reset_index()
                tempps = tempps.loc[tempps.prediction==cat]
                rawps= tempps.apply( lambda x: 1 if x.predictive_status=='TP' else 0,axis=1)
                rawps = rawps.tolist()
                ps = torch.Tensor(rawps).long()
            return ps
        except Exception as e:
            print("Error {}".format(e))
            return torch.Tensor([])

    def make_omicron_test(self,n_class:int,tpr_threshold=.95):
        """Outputs the corresponding logistic omicron and competence based test calibrated 
        to the True Positive Rate threshold.

        Parameters
        -----------------
        :self.omicrons: This would be the internal omicron dictionary
        :tpr_threshold: float. Default: .95. Increasing tpr_threshold increases FP
        :n_class: number of separate classes to range over, even if unobserved. In such cases, test
        defaults to 0 values.
        """
        omicron_d = self.omicrons
        omicrontestparams= dict({'n_class':n_class,
                                 'predictive_status':'Global',
                                 'tpr_threshold':tpr_threshold,
                                })
        return generate_omicron_test(omicron_d,omicrontestparams)

    def _get_certainty_sample(self,
                              certainty_dist:pd.DataFrame,
                              omicrons: torch.Tensor,
                              certainties:Certainties,
                              cat:Union[int,str],
                              predictive_status:Optional[str] = None,
                              sample_size_max = 50,
                              is_bayesian = False):
        """ Given the category to select, the corresponding predictive_status if anything to control for,
        and finally the maximum sample size to sample from, we look at the distribution of the omicrons 
        from least to greatest in appropriate sample size increments, and then find the corresponding certainty 
        in the self.data['certainty'] or self.data['ensemble_certainty'] object that corresponds to the observed
        omicron. 

        Parameters
        ----------------------
        :certainty_dist: pd.DataFrame; ideally extracted from the certainty_distribution
        :omicrons: the underlying omicrons corresponding to the conditioned/predicted category, that we use to find the sample from. 
        :certainties: an instance of the Certainties object corresponding to the provided certainty dist
        :cat: int or str, used to identity the category to condition the predicted certainties on when computing the omicrons
        :predictive_status: optional str, should either be 'TP' or 'FP', or left as None, indicating how to condition the certainty score distributions
        :sample_size_max: int, default=50, indicates the maximum number of samples of certainties to draw per category
        :is_bayesian: bool, default=False, indicates if the underlying data is drawn from a Bayesian sample
        """
        tempdf = certainty_dist.copy() #reset_index()
        if predictive_status is None:
            select_index = tempdf.loc[(tempdf.prediction == cat)].index.tolist()
        else:
            select_index = tempdf.loc[(tempdf.prediction == cat) & (tempdf.predictive_status == predictive_status)].index.tolist()
        # now we have the corresponding indices to select our certainties from, we now need to find the corresponding sample from our omicrons
        # This corresponding sample will be the relative index from the select_index
        sample_size = min(sample_size_max, len(select_index), omicrons.shape[0])
        if sample_size>0:
            increment = (1/sample_size)
            sample_indices = []
            quantile_pos = 0
            while (quantile_pos <1):
                omicron_q=torch.quantile(omicrons,q = quantile_pos, interpolation = "lower")
                sample_indices.append( (omicrons == omicron_q).nonzero()[0].item())
                quantile_pos += increment
            sampled_indices = [select_index[n] for n in sample_indices]
            if is_bayesian:
                sampled_certainties = certainties.ensemble_certainty[sampled_indices]
            else:
                sampled_certainties = certainties.certainty[sampled_indices]
        else:
            sampled_certainties = None
        return sampled_certainties
    
    def to_dict(self):
        """ Convert instance attributes to a dictionary."""
        return self.__dict__
         
    def save(self,address:str,name:Optional[str]=None):
        """ Save a pickled representing the attributes of the Omicron_data object"""
        dict_to_save = self.to_dict()
        if name:
            omicron_name = name+".pickle"
        else:
            omicron_name = "_omicrons.pickle"
        if pickle:
            with open(os.path.join(address,omicron_name),'wb') as handle:
                pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self,address:str,name:Optional[str]=None):
        """  Load in the saved representation of the omicron data. """
        try:
            if name:
                omicron_name = name+".pickle"
            else:
                omicron_name = "_omicrons.pickle"
            with open(os.path.join(address,omicron_name),'rb') as file:
                omicron_data = pickle.load(file) 
            for name,attr in omicron_data.items():
                setattr(self,name,attr)
        except Exception as E:
            print("Failed to load due to the following error:\n {}".format(E))

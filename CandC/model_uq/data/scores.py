# scores.py

import pandas as pd
import numpy as np
import os
import pickle

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    average_precision_score,
    precision_score,
)

from typing import Union, Optional


from ...oodd.mwu import  *
from ...oodd.pcs_vro import *

from ...candc.competence import (
    component_competence,
    empirical_competence,
)
from ...certainty_stats import *

from .gather import *
from .model_data import Output_Data, Model_Data
from .assignmentdf import Assignment_DF
from .certainties import Certainties
from .certainty_distribution import Certainty_Distribution


class Scores():
    def __init__(self,
                 address:Optional[str]=None,
                 name:Optional[str]=None,
                 tpr_threshold=0.95):
        """Data object collecting various uncertainty scores, drawn from classical
        confusion matrix analysis, as well as the certainty and competence framework.

        Initialization without providing both an address and name of score file will fail
        to load, and instead an empty dictionary will be initialized.

        To fill the dictionary, run the various methods for scores, with get_bin_scores
        producing the 
        """
        if isinstance(address,str):
            try:
                if name is not None:
                    scores_name = name
                else:
                    scores_name = "_scores"
                self.load(address=address,name=scores_name)
            except Exception as E:
                print("Since {}, we will fill in scores from scratch".format(E))
                self.scores=dict()
                self.tpr_threshold=tpr_threshold
        else:
            self.scores = dict()
            self.tpr_threshold = tpr_threshold 
    
    def load_in_assignment_df(self,
                              assignment_df_address:Optional[str]=None,
                              assignment_df_name:Optional[str]=None,
                              assignment_df: Optional[Assignment_DF]=None,
                             ):
        """Load in model data from provided address and name, or otherwise pass model_data as a parameter.

        Parameters
        -----------------------
        :model_data_address: provides directory to find model data
        :model_data_name: provides optional name to load in specific model data object
        :model_data: model_data object to pass
        """
        try:
            if isinstance(assignment_df,Assignment_DF):
                print("Passing assignment_df")
                return assignment_df
            else:
                print("Attempting to load assignment_df saved at {}".format(assignment_df_address))
                if assignment_df_address is not None: 
                    if assignment_df_name is None:
                        assignment_df_name = "_assignment_df"
                    assignment_df = Assignment_DF(assignment_df_address=assignment_df_address,assignment_df_name=assignment_df_name,load=True)
                    return assignment_df
                else:
                    raise ValueError("Empty values provided for model_data_address and model_data. Pick one.")
        except Exception as E:
            print("While loading in assignment dataframe, the following error occurred:\n {}".format(E))
                        
    def load_in_model_data(self,
                           model_data_address:Optional[str]=None,
                           model_data_name:Optional[str]=None,
                           model_data: Optional[Model_Data]=None):
        """ Load in model data from provided address and name, or otherwise pass model_data as a parameter.

        Parameters
        -----------------------
        :model_data_address: provides directory to find model data
        :model_data_name: provides optional name to load in specific model data object
        :model_data: model_data object to pass
        """
        try:
            if model_data:
                return model_data
            else:
                if model_data_address is not None:
                    if model_data_name:
                        model_data_name = model_data_name
                    else:
                        model_data_name = "_model_data"
                    model_data = Model_Data(model_data_address=model_data_address,model_data_name=model_data_name)
                    return model_data
                else:
                    raise ValueError("Empty values provided for model_data_address and model_data. Pick one.")
        except Exception as E: #except Exception as E:
            print("While loading in model_data, the following error occurred:\n {}".format(E))

    def load_in_certainty_dist(self,
                               certainty_dist_address:Optional[str]=None,
                               certainty_dist_name:Optional[str]=None,
                               certainty_dist: Optional[Certainty_Distribution]=None):
        """ Load in Certainty Distribution from provided address and name, or otherwise pass model_data as a parameter.

        Parameters
        -----------------------
        :certainty_dist_address: provides directory to find Certainty Distribution
        :certainty_dist_name: provides optional name to load in specific Certainty_Distribution object
        :certainty_dist: Certainty Distribution object to pass
        """
        try:
            if certainty_dist is not None:
                return certainty_dist
            else:
                if certainty_dist_address is not None:
                    if certainty_dist_name is not None:
                        certainty_dist_name = certainty_dist_name
                    else:
                        certainty_dist_name = "_certainty_dist"
                    certainty_dist = Certainty_Distribution(address=certainty_dist_address,name=certainty_dist_name)
                    return certainty_dist
                else:
                    raise ValueError("Empty values provided for certainty distribution address and certainty_dist. Pick one.")
        except Exception as E:
            print("While loading in certainty distribution, the following error occurred:\n {}".format(E))

    def load_in_certainties(self,
                               certainties_address:Optional[str]=None,
                               certainties_name:Optional[str]=None,
                               certainties: Optional[Certainty_Distribution]=None):
        """ Load in Certainty Distribution from provided address and name, or otherwise pass model_data as a parameter.

        Parameters
        -----------------------
        :certainties_address: provides directory to find Certainties object
        :certainties_name: provides optional name to load in specific Certainties object
        :certainties: Certaintainties object to pass
        """
        try:
            if certainties:
                return certainties
            else:
                if certainties_address is not None:
                    if certainties_name is not None:
                        cert_name = certainties_name
                    else:
                        cert_name = "_certainties"
                    certainties = Certainties(address=certainties_address,name=cert_name)
                    return certainties
                else:
                    raise ValueError("Empty values provided for certainties address and certainties. Pick one.")
        except Exception as E:
            print("While loading in certainties, the following error occurred:\n {}".format(E))
    
    def get_bin_scores(self,
                       assignment_df_address : Optional[str]=None,
                       assignment_df_name : Optional[str]=None,
                       assignment_df:Optional[pd.DataFrame]=None,
                       model_data_address:Optional[str]=None,
                       model_data_name:Optional[str]=None,
                       model_data:Optional[Union[Output_Data,Model_Data]]=None,
                       certainties_address : Optional[str]=None,
                       certainties_name  : Optional[str]=None,
                       certainties: Optional[Certainties]=None,
                       is_bayesian=False):
        """ Give the dataframe saved in the Assignment_DF, and an output_data object 
        (either in the form of the Output_Data class or the Model_Data class),  
        get_bin_scores computes the 'binary' confusion matrix scores by collapsing all 
        non-safe values to a single category. Then find the following values
        :condition positive (P): the number of real 'positive' cases
        :condition negative (N): the number real 'negative'/benign/safe cases in the data
        :misclassified positive (MP): the number of real, but incorrectly classified
        positive cases in data
        :true positive (TP): a test result that correctly indicates the presence of a condition
        :true negative (TN): a test result that correctly indicates the absence of a condition
        :false positive (FP): a test result which wrongly indicates that a particular
        condition is present
        :false negative (FN): a test result which wrongly indicates that a particular
        condition is absent
        :sensitivity (TPR): TP/P
        :specificity (TNR): TN/N
        :precision (PPV): TP/(TP+FP)
        :negative predictive value (NPV): TN/(TN+FN)
        :miss rate (FNR): FN/P
        :fall-out (FPR): FP/N
        :false discovery rate (FDR): 1-PPV
        :false omission rate (FOR): 1-NPV
        :Positive Likelihood ratio (LR+): TPR/FPR
        :negative likelihood ratio (LR-): FNR/TNR
        :prevalence threshhold (PT): sqrt(FPR)/(sqrt(TPR)+sqrt(FPR))
        :threat score (TS): TP/(TP+FN+FP)
        :prevalence: P/(P+N)
        :accuracy (ACC): (TP+TN)/(P+N)
        :balance accuracy (BA): (TPR+TNR)/2
        :F1 score: from  scipy.stats
        :BF1 score: 2TP/(2TP+FP+FN)
        :phi coefficient (MCC): (TPxTN - FPxFN)/sqrt((TP+FP)(TP+TN)(TN+FP)(TN+FN))
        :Fowlkes-Mallows Index (FM): sqrt(PPVxTPR)
        :informedness (BM): TPR+TNR-1
        :markedness (MK): PPV+NPV-1
        :diagnostic odds ratio(DOR): LR+/LR-

        Afterwards, will attempt to run get_remaining_scores

        Parameters
        ---------------------------------
        :assignment_df_address : Optional[str]=None,
        :assignment_df_name : Optional[str]=None,
        :assignment_df:Optional[pd.DataFrame]=None,
        :model_data_address:Optional[str]=None,
        :model_data_name:Optional[str]=None,
        :model_data:Optional[Union[model_data,Model_Data]]=None,
        :is_bayesian=False
        """
        df = self.load_in_assignment_df(assignment_df_address=assignment_df_address,
                                        assignment_df_name=assignment_df_name,
                                        assignment_df=assignment_df,
                                       )
        df = df.data
        model_data = self.load_in_model_data(model_data_address=model_data_address,
                                              model_data_name=model_data_name,
                                              model_data=model_data)
        certainties = self.load_in_certainties(certainties_address = certainties_address,
                                               certainties_name = certainties_name,
                                               certainties = certainties)
        self.scores.__setitem__("P",
                                len(df.loc[~df.classification.isin(model_data.safevalues)]))
        self.scores.__setitem__("N",
                                len(df.loc[df.classification.isin(model_data.safevalues)]))
        self.scores.__setitem__("MP",
                                len(df.loc[~(df.classification.isin(model_data.safevalues))&(df.prediction!=df.classification)]))
        self.scores.__setitem__("TP",
                                len(df.loc[~(df.classification.isin(model_data.safevalues))&~(df.prediction.isin(model_data.safevalues))]))
        self.scores.__setitem__("TN",
                                len(df.loc[(df.classification.isin( model_data.safevalues))&(df.prediction.isin(model_data.safevalues))]))
        self.scores.__setitem__("FP",
                                len(df.loc[(df.classification.isin( model_data.safevalues))&~(df.prediction.isin(model_data.safevalues))]))
        self.scores.__setitem__("FN",
                                len(df.loc[~(df.classification.isin( model_data.safevalues))&(df.prediction.isin(model_data.safevalues))]))
        self.scores.__setitem__("TPR",
                                self.scores['TP']/self.scores['P'] if (self.scores['P'])>0 else None)
        self.scores.__setitem__("TNR",
                                self.scores['TN']/self.scores['N'] if (self.scores['N'])>0 else None)
        self.scores.__setitem__("ER",
                                (self.scores['MP']+ self.scores['FP']+ self.scores['FN'])/(self.scores['P']+ self.scores['N']) if (self.scores['P']+ self.scores['N'])>0 else None)
        self.scores.__setitem__("MPR",
                                self.scores['MP']/self.scores['P']if (self.scores['P'])>0 else None)
        self.scores.__setitem__("PPV",
                                self.scores['TP']/(self.scores['TP']+ self.scores['FP']) if (self.scores['TP']+ self.scores['FP'])>0 else None)
        self.scores.__setitem__("NPV",
                                self.scores['TN']/(self.scores['TN']+ self.scores['FN']) if (self.scores['TN']+ self.scores['FN'])>0 else None)
        self.scores.__setitem__("FNR",
                                self.scores['FP']/self.scores['P'] if (self.scores['P'])>0 else None)
        self.scores.__setitem__("FPR",
                                self.scores['FP']/self.scores['N']if (self.scores['N'])>0 else None)
        self.scores.__setitem__("FDR",
                                self.scores['FP']/(self.scores['FP']+ self.scores['TP']) if (self.scores['TP']+ self.scores['FP'])>0 else None)
        self.scores.__setitem__("FOR",
                                self.scores['FN']/(self.scores['FN']+ self.scores['TN'])if (self.scores['TN']+ self.scores['FN'])>0 else None)
        self.scores.__setitem__("CER",
                                self.scores['FP']/(self.scores['MP']+ self.scores['FN'])if (self.scores['MP']+ self.scores['FN'])>0 else None)
        try:
            self.scores.__setitem__("LR+",
                                    self.scores['TPR']/self.scores['FPR'] if (self.scores['FPR'] !=None) and (self.scores['FPR'] >0.0) else None)
        except Exception as e:
            print(e)
        try:
            self.scores.__setitem__("LR-",
                                    self.scores['FNR']/self.scores['TNR'] if (self.scores['TNR'] !=None ) and (self.scores['TNR'] >0.0 )else None)
        except Exception as e:
            print(e)
        try:
            self.scores.__setitem__("PT",
                                    math.sqrt(self.scores['FPR'])/(math.sqrt(self.scores['TPR'])+math.sqrt(self.scores['FPR']))if (self.scores['FPR']!=None )and (self.scores['TPR']!=None )else None)
        except Exception as e:
            print(e)
        try:
            self.scores.__setitem__("TS",
                                    self.scores['TP']/(self.scores['TP']+ self.scores['FN']+ self.scores['FP']) if( (self.scores['TP']+ self.scores['FN']+ self.scores['FP'])>0.0) else None)
        except Exception as e:
            print(e)
        try:
            self.scores.__setitem__("Prevalence",
                                    self.scores['P']/(self.scores['P']+ self.scores['N']))
        except Exception as e:
            print(e)
        try:
            self.scores.__setitem__("MCA",
                                    df.apply(lambda x: 1 if x.classification==x.prediction else 0,axis=1).mean())
        except Exception as e:
            print(e)
        try:
            self.scores.__setitem__("BCA",
                                    (self.scores['TP']+ self.scores['TN'])/(self.scores['P']+ self.scores['N']))
        except Exception as e:
            print(e)
        try:
            self.scores.__setitem__("BAL",
                                    (self.scores['TPR']+ self.scores['TNR'])/2 if( self.scores['TPR']!=None) and (self.scores['TNR']!=None )else None)
        except Exception as e:
            print(e)
        try:
            self.scores.__setitem__("F1",
                                    f1_score(df.classification,df.prediction,
                                              labels=np.array(list(model_data.classification_categories.keys())),
                                              average=None,
                                              zero_division=0))
        except Exception as e:
            print(e)
        try:
            self.scores.__setitem__("BF1",
                                    (2*self.scores['TP'])/(2*self.scores['TP']+ self.scores['FP']+ self.scores['FN']) if (2*self.scores['TP']+ self.scores['FP']+ self.scores['FN'])>0.0 else None )
        except Exception as e:
            print(e)
        try:
            self.scores.__setitem__("MCC",
                                    (self.scores['TP']*self.scores['TN']-self.scores['FP']*self.scores['FN'])/(math.sqrt((self.scores['TP']+ self.scores['FP'])*(self.scores['TP']+ self.scores['FN'])*(self.scores['TN']+ self.scores['FP'])*(self.scores['TN']+ self.scores['FN']))) if (math.sqrt((self.scores['TP']+ self.scores['FP'])*(self.scores['TP']+ self.scores['FN'])*(self.scores['TN']+ self.scores['FP'])*(self.scores['TN']+ self.scores['FN'])))>0.0 else None)
        except Exception as e:
            print(e)
        try:
            self.scores.__setitem__("FM",
                                    math.sqrt(self.scores['PPV']*self.scores['TPR']) if self.scores['PPV']!=None and self.scores['TPR']!=None else None)
        except Exception as e:
            print(e)
        try:
            self.scores.__setitem__("BM",
                                    self.scores['TPR']+ self.scores['TNR']-1 if self.scores['TPR']!= None and self.scores['TNR']!=None else None)
        except Exception as e:
            print(e)
        try:
            self.scores.__setitem__("MK",
                                    self.scores['PPV']+ self.scores['NPV']-1 if self.scores['PPV']!=None and self.scores['NPV']!=None else None)
        except Exception as e:
            print(e)
        try:
            self.scores.__setitem__("DOR",
                                    self.scores['LR+']/self.scores['LR-'] if self.scores['LR+']!=None and self.scores['LR-']!=None else None)
        except Exception as e:
            print(e)
        try:
            self.get_remaining_scores(assignment_df=df,
                                      model_data=model_data,
                                      certainties=certainties,
                                      is_bayesian=is_bayesian)
        except Exception as e:
            print(e)
            
    def get_scores(self,
                   assignment_df_address : Optional[str]=None,
                   assignment_df_name : Optional[str]=None,
                   assignment_df:Optional[pd.DataFrame]=None,
                   model_data_address:Optional[str]=None,
                   model_data_name:Optional[str]=None,
                   model_data:Optional[Union[Output_Data,Model_Data]]=None,
                   certainties_address : Optional[str]=None,
                   certainties_name  : Optional[str]=None,
                   certainties: Optional[Certainties]=None,
                   is_bayesian=False):
        """ Given an Assignment_DF dataframe as input df, will compute:
        :multiclass accuracy: either with respect to 'predictions'
        (currently under development in the Assignment DF object), or predictions category;
        :TP: the number of true positives found in the Assignment_DF object
        :FP: the number of false positives found in the Assignment_DF object
        :F1: the F1 scores with average=None, and zero_division=0

        before running get_remaining_scores
        """
        assignment_df = self.load_in_assignment_df(assignment_df_address=assignment_df_address,
                                                   assignment_df_name=assignment_df_name,
                                                   assignment_df=assignment_df_name)
        model_data = self.load_in_model_data(model_data_address=model_data_address,
                                             model_data_name=model_data_name,
                                             model_data= model_data)
        certainties = self.load_in_certainties(certainties_address = certainties_address,
                                               certainties_name = certainties_name,
                                               certainties = certainties)
        n= len(assignment_df.data)
        if 'predictions' in assignment_df.data.columns.tolist():
            self.scores.__setitem__("MCA",len(assignment_df.data[assignment_df.data.apply(lambda x: x.classification in x.predictions,axis=1)])/n)
        else:
            self.scores.__setitem__("MCA",len(assignment_df.data[assignment_df.data.apply(lambda x: x.classification in x.predictions,axis=1)])/n)
        self.scores.__setitem__("TP",len(assignment_df.data.loc[assignment_df.data.predictive_status=='TP']))
        self.scores.__setitem__("FP",len(assignment_df.data.loc[assignment_df.data.predictive_status=='FP']))
        tempdf= pd.DataFrame({'classification':assignment_df.data.classification,
                               'prediction':assignment_df.data.prediction
                              })
        tempdf=tempdf.dropna()
        self.scores.__setitem__("F1",f1_score(tempdf.classification,tempdf.prediction,
                                              labels=np.array(list(model_data.classification_categories.keys())),
                                              average=None,
                                              zero_division=0))
        self.get_remaining_scores(assignment_df=assignment_df,
                                  model_data=model_data,
                                  certainties=certainties,
                                  is_bayesian=is_bayesian)

    def get_remaining_scores(self,
                             assignment_df:Assignment_DF,
                             model_data:Model_Data,
                             certainties:Certainties,
                             is_bayesian=False):
        """ Under development. Presently we also attempt to gather:
        :Average Precision Score (APS):
        :Area Under ROC (RAS):
        :Cohen's Kappa (CK):
        :Predictive Entropy (PE):
        :Mutual Information Prediciton (MIP):
        :Component Competence (CC):
        :Empirical Competence (EC):
        """
        remaining_scores=['APS','RAS','CK','PE','MIP','CC','EC']
        for score in remaining_scores:
            try:
                self.scores.__setitem__(score, self._gather_score(name=score,
                                                                  assignment_df=assignment_df,
                                                                  model_data=model_data,
                                                                 certainties=certainties,))
            except ValueError as e:
                print('While trying to gather {}, ran into Value Error {}'.format(score,e))

    def _gather_score(self,
                      name:str,
                      assignment_df:Optional[Assignment_DF]=None,
                      model_data:Optional[Model_Data]=None,
                      certainties:Optional[Certainties]=None,
                      is_bayesian=False):
        """
        Gathers the named score (name) from assignment_df and model_data.
        """
        print('Gathering {}'.format(name))
        if name=='APS':
            y_test=np.array(assignment_df.data.classification)
            if is_bayesian:
                return precision_score(y_true= y_test,
                                       y_pred=model_data.ensemble_prediction.detach().numpy(),
                                       pos_label=None,
                                       average='weighted',
                                       zero_division=np.nan)
            else:
                return precision_score(y_true=y_test,
                                       y_pred=np.array(model_data.prediction.detach().numpy()),
                                       pos_label=None,
                                       average='weighted',
                                       zero_division=np.nan)
        if name=='RAS':
            if is_bayesian:
                ref ='ensemble_output'
            else:
                ref ='output'
            return roc_auc_score(y_true=assignment_df.data.classification,
                                 y_score=getattr(model_data,ref).detach().numpy(),
                                 multi_class='ovr',
                                 labels= list(model_data.classification_categories.keys()))
        if name=='CK':
            if is_bayesian:
                return cohen_kappa_score(model_data.input_data_labeled.values,
                                         model_data.ensemble_prediction.detach().numpy())
            else:
                return cohen_kappa_score(assignment_df.data.classification,
                                         model_data.prediction.detach().numpy())
        if name=='PE':
            if is_bayesian:
                return predictive_entropy(model_data.output.detach().numpy())
            else:
                return None

        if name=='MIP':
            if is_bayesian:
                return mutual_information_prediction(model_data.output)
            else:
                return None
        
        if name =='CC':
            if is_bayesian:
                return component_competence(predicted_prob=torch.Tensor(model_data.ensemble_output),
                                            observed_label=torch.Tensor(model_data.input_data_labeled.numpy()).to(int).reshape(-1))
            else:
                return component_competence(predicted_prob=torch.Tensor(model_data.output),
                                            observed_label=torch.Tensor(model_data.input_data_labeled.numpy()).to(int).reshape(-1))
        
        if name =='EC':
            if is_bayesian:            
                return empirical_competence(
                    find_certainty_dist_dataframe(**dict({'classification':assignmenty_df.data.classification,
                                                     'cat_predict':certainties.ensemble_predictions,
                                                     'cert_score':certainties.ensemble_certainty_score,
                                                     'is_bayesian':is_bayesian}))
                )
            else:
                return empirical_competence(
                    find_certainty_dist_dataframe(
                        **dict({'classification':assignment_df.data.classification,
                                'cat_predict':certainties.predictions,
                                'cert_score':certainties.certainty_score,
                               'is_bayesian':is_bayesian})
                    )
                )
                
    def gather_certainty_score_stats(self,
                                     total_gatherlist:list,
                                     predictive_comparison_gatherlist:list,
                                     assignment_df_address:Optional[str]=None,
                                     assignment_df_name:Optional[str]=None,
                                     assignment_df:Optional[Assignment_DF]=None,
                                     certainties_address:Optional[str]=None,
                                     certainties_name:Optional[str]=None,
                                     certainties: Optional[Certainties]=None,
                                     is_bayesian=False):
        assignment_df = self.load_in_assignment_df(assignment_df_address = assignment_df_address,
                                                   assignment_df_name = assignment_df_name,
                                                   assignment_df = assignment_df)
        data = self.load_in_certainties(certainties_address=certainties_address,
                                        certainties_name=certainties_name,
                                        certainties=certainties)
        print('Gathering Certainty Score stats')
        if is_bayesian:
            ref,func = 'ensemble_certainty_score', self._fill_certainty_score_stats
            frame = getattr(data,ref)
            if frame is not None:
                print('\t -Gathering in sample certainty scores')
                setattr(self,'in_sample_cert_scores', gather_for_all(func,assignment_df.data,False,*total_gatherlist,**dict({'frame':frame.detach().clone().numpy()})) ) 
                print('\t -Gathering MWU Results')
                ref,func ='ensemble_certainty_score',mwu_certainty_dist_test_internal
                self.in_sample_cert_scores.__setitem__('MWU Results',
                                                             gather_for_all_with_predictive_comparison(func,assignment_df.data,*predictive_comparison_gatherlist,**{'frame':frame.detach().clone().numpy()}))

        else:
            ref,func='certainty_score',self._fill_certainty_score_stats
            frame = getattr(data,ref)
            print('\t -Gathering in sample certainty scores')
            if frame is not None:
                setattr(self,'in_sample_cert_scores',
                                        gather_for_all(func,assignment_df.data,False,*total_gatherlist,**dict({'frame':frame.detach().clone().numpy()})))
                ref,func='certainty_score',mwu_certainty_dist_test_internal
                print('\t -Gathering MWU Results')
                self.in_sample_cert_scores.__setitem__('MWU Results',
                                                                 gather_for_all_with_predictive_comparison(func,assignment_df.data,*predictive_comparison_gatherlist,**dict({'frame':frame.detach().clone().numpy()})))
        del data
            
    def _fill_certainty_score_stats(self,reference_frame):
        local_dict = dict()
        local_dict.__setitem__('certainty_scores',reference_frame)
        local_dict.__setitem__('mean' , local_dict['certainty_scores'].mean())
        local_dict.__setitem__('median', np.median(local_dict['certainty_scores']))
        local_dict.__setitem__('std',local_dict['certainty_scores'].std())
        return local_dict
        
    def get_empirical_competencies(self,
                                   classification_categories:dict[int,Union[str,int]],
                                   certainty_dist_address:Optional[str]=None,
                                   certainty_dist_name:Optional[str]=None,
                                   certainty_dist:Optional[Certainty_Distribution]=None,
                                   ):
        """Given the certainty_distribution DataFrame, and the classification_categories 
        dictionary object from the Input_Data or Model_Data object, creates a dictionary
        object recording the observed empirical competences for each category.
        """
        if True:#try:
            certainty_dist = self.load_in_certainty_dist(certainty_dist_address=certainty_dist_address,
                                                         certainty_dist_name=certainty_dist_name,
                                                         certainty_dist=certainty_dist)
            ec_dict=dict({})
            tempdf = certainty_dist.data.reset_index()
            for cat in classification_categories.keys():
                select_indices = tempdf.loc[(tempdf.prediction==cat)].index.tolist()
                subtempdf = tempdf.iloc[select_indices]
                ec_dict.__setitem__(cat,empirical_competence(subtempdf))
            setattr(self,'empirical_competencies',ec_dict)
        if False:#except Exception as E:
            print("Following error was raised:\n {}".format(E))

    def get_component_competencies(self,
                                   classification_categories:dict[int,Union[str,int]],
                                   model_data_address:Optional[str]=None,
                                   model_data_name:Optional[str]=None,
                                   model_data:Optional[Model_Data]=None,
                                   is_bayesian=False):
        """Given the certainty_distribution DataFrame, and the classification_categories 
        dictionary object from the Input_Data or Model_Data object, creates a dictionary
        object recording the observed component competences for each category.
        """
        if True:#try:
            model_data = self.load_in_model_data(model_data_address = model_data_address,
                                                 model_data_name = model_data_name,
                                                 model_data = model_data)
            cc_dict=dict({})
            tempdf = pd.DataFrame(model_data.input_data_labeled.numpy(), columns=['prediction']).reset_index()
            for cat in classification_categories.keys():
                select_indices = torch.Tensor(tempdf.loc[(tempdf.prediction==cat)].index.tolist()).long()
                if is_bayesian:
                    subtempdf= component_competence(predicted_prob=torch.Tensor(model_data.ensemble_output[:,select_indices,:]),
                                        observed_label=torch.Tensor(model_data.input_data_labeled.numpy()).to(int).reshape(-1)[select_indices])
                else:
                    subtempdf =  component_competence(predicted_prob=torch.Tensor(model_data.output[select_indices,:]), 
                                                      observed_label=torch.Tensor(model_data.input_data_labeled.numpy()).to(int).reshape(-1)[select_indices])
                cc_dict.__setitem__(cat,subtempdf)
            setattr(self,'component_competencies',cc_dict)
        if  False:#Except Exception as E:
            print("While getting component competencies, the following error arose:\n{}".format(E))
            
    def gather_pcs_vro_stats(self,
                             classification_gatherlist:list,
                             certainties_address:Optional[str]=None,
                             certainties_name:Optional[str]=None,
                             certainties:Optional[Certainties]=None,
                             return_pcs_vro_test = True,
                            ):
        """
        Gather the PCS-VRO statistics per the required classification gather list, and return the corresponding
        pcs-vro test if selected

        Parameters
        ----------------
        :classification_gatherlist: list indicating classification status and category (optional)
        :certainties_address: address to load in certainties object
        :certainties_name: additional name to load in specific certainties object
        :certainties: optional certainties object to use in lieu of loading in a certainties object
        :return_pcs_vro_test: bool, set to True in order to return a pcs_vro test from the certainties data
        
        """
        func=self._gather_pcs_vro_stats
        stats,tests= self.gather_stats_and_tests_for_all_from_two_frames(func,*classification_gatherlist,**{'frames':(certainties.ensemble_certainty_score,certainties.VRO)})
        self.scores.__setitem__('PCS-VRO stats',stats)
        self.oodd_tests.__setitem__('PCS-VRO Test',tests)
        print("Added 'PCS-VRO stats' to scores, with keys {}".format(stats.keys()))
        print("Added 'PCS-VRO Test' to oodd_tests, with keys {}".format(tests.keys()))
    
    def _gather_pcs_vro_stats(self,reference_frame_one,reference_frame_two):
        local_tests=dict({})
        local_stats=dict({})
        HPCS,LPCS,HVRO,LVRO=pcs_vro_fit(indata_pcs=reference_frame_one,
                                        indata_vro=reference_frame_two,
                                        tpr_threshold=self.tpr_threshold)
        local_tests.__setitem__('HPCS',HPCS)
        local_tests.__setitem__('LPCS',LPCS)
        local_tests.__setitem__('HVRO',HVRO)
        local_tests.__setitem__('LVRO',LVRO)
        local_tests.__setitem__('PCS-VRO Test',
                               pcs_vro_test(pcs=reference_frame_one,
                                            vro=reference_frame_two,
                                            HPCS=local_tests['HPCS'],
                                            LPCS=local_tests['LPCS'],
                                            HVRO=local_tests['HVRO'],
                                            LVRO=local_tests['LVRO']))
        local_stats.__setitem__('PCS-VRO Test Summary',
                               pcs_vro_test_summary(pcs=reference_frame_one,
                                            vro=reference_frame_two,
                                            HPCS=local_tests['HPCS'],
                                            LPCS=local_tests['LPCS'],
                                            HVRO=local_tests['HVRO'],
                                            LVRO=local_tests['LVRO']))
        local_stats.__setitem__('Mean PCS', reference_frame_one.mean())
        local_stats.__setitem__('Mean VRO', reference_frame_two.mean())
        return local_stats,local_tests
        
    def to_dict(self):
        """ Convert instance attributes to a dictionary."""
        return self.__dict__
        
    def save(self,
             address:str,
             name:Optional[str]=None):
        """ Save a pickled representing the attributes of the Certainties object"""
        dict_to_save = self.to_dict()
        if name:
            scores_name = name+".pickle"
        else:
            scores_name = "_scores.pickle"
        with open(os.path.join(address,scores_name),'wb') as handle:
            pickle.dump(dict_to_save, handle,protocol=pickle.HIGHEST_PROTOCOL)

    def load(self,
             address:str,
             name:Optional[str]=None):
        if name:
            scores_name = name+".pickle"
        else:
            scores_name = "_scores.pickle"
        with open(os.path.join(address,name), 'rb') as file:
            score_dict = pickle.load(file)
        setattr(self,'scores',score_dict)
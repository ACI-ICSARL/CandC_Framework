# assignmentdf.py
import pandas as pd
import numpy as np
import torch
import pickle
import os
from typing import Optional, Union

class Assignment_DF():
    f""" The Assignment_DF object is a particular form of the pandas dataframe the collects 
    the relevant information for producing objects in the CandC_Framework.
    """
    def __init__(self,
                 assignment_df_address:Optional[str]=None,
                 assignment_df_name:Optional[str]=None,
                 prediction: Optional[Union[pd.Series,np.ndarray,torch.Tensor]]=None,
                 classification: Optional[Union[pd.Series,np.ndarray,torch.Tensor]]=None,
                 predictions: Optional[Union[list[pd.Series],list[np.ndarray],list[torch.Tensor]]]=None,
                 predictive_status: Optional[Union[pd.Series,list[str]]]=None,
                 assignment_df: Optional[pd.DataFrame]=None,
                 load=False):
        # first, trying loading from file folder and name if provided, else check if assignment_df is empty or not
        try:
            if load:
                if isinstance(assignment_df_address,str):
                    self.load(address=assignment_df_address,
                              name=assignment_df_name)
            else:
                if isinstance(assignment_df,pd.DataFrame):
                    cleaned_data = self._validate_dataframe(assignment_df)       
                else:
                    cleaned_data = self._validate_data(prediction,
                                                       classification,
                                                       predictions,
                                                       predictive_status)
                self.data = cleaned_data
        except Exception as e:
            print("While trying to initialize an Assignment_DF, we raised the following error: {}".format(e))
    
    def _validate_dataframe(self, df:pd.DataFrame):
        """  Check if the dataframe given is correctly structured for our purposes. If the data
        needs to be cleaned, we make attempts to do so, and otherwise raise an error if there is information
        missing.
        """
        if 'prediction' not in df.columns.tolist():
            raise ValueError("Provided dataframe missing prediction column")
        else:
            if df.prediction.dtype!='int':
                df.prediction.astype(int)
        if 'predictions' not in df.columns.tolist():
            raise ValueError("Provided dataframe missing predictions column")
        else:
            if not df.predictions.apply(lambda x: isinstance(x,list) and all(isinstance( i,int) for i in x)).all():
                raise ValueError("The provided predictions are not all integer values. Please fix")

        if 'classification' in df.columns.tolist():
            if df.classification.dtype!='int':
                df.classification.astype(int)
            if 'predictive_status' not in df.columns.tolist():
                df['predictive_status']=df.apply(lambda x: 'TP' if x.classification in x.predictions else 'FP',axis=1)
        else:
            df['classification'] = df.apply(lambda x: int(-1), axis=1)
            df['predictive_status'] = df.apply(lambda x: 'U', axis=1)
        data = df[['prediction','predictions','classification','predictive_status']]
        return data
        
    def _validate_data(self,
                       prediction:Union[pd.Series,np.ndarray,torch.Tensor],
                       classification:Optional[Union[pd.Series,np.ndarray,torch.Tensor]]=None,
                       predictions:Optional[Union[list[pd.Series],list[np.ndarray],list[torch.Tensor]]]=None,
                       predictive_status: Optional[Union[pd.Series,list[str]]]=None):
        """ Checks that that the given prediction(s) and classifications are of the correct type
        and creates valid dataframe object.

        Parameters
        --------------------
        :prediction: 1-d presentation of a model's predicted category
        :classification: optional presentation containing ground truth labeling 
        :predictions: optional presentation containing additional predictions
        :predictive_status: optional 1-d presentation indicating if the data is TP, FP, or unknown
        """
        # Handle the prediction argument
        try:
            if isinstance(prediction,torch.Tensor):
                prediction = prediction.detach().cpu().numpy()
            if isinstance(prediction,pd.Series):
                if prediction.dtype !='int':
                    prediction=prediction.astype(int)
            if isinstance(prediction,np.ndarray):
                if prediction.dtype !='int':
                    prediction = prediction.astype(int)
            # Handle the classification argument    
            if isinstance(classification,torch.Tensor):
                classification = classification.detach().cpu().numpy()
            if isinstance(classification,pd.Series):
                if classification.dtype !='int':
                    classification=classification.astype(int)
            if isinstance(classification,np.ndarray):
                if classification.dtype !='int':
                    classification = classification.astype(int)
            # confirm they are the same length        
            if len(prediction)!=len(classification):
                raise ValueError("Size mismatch between prediction and classification. {} != {}".format(
                    len(prediction),len(classification)))
            # Now we handle predictions
            if isinstance(predictions,list):
                if isinstance(predictions[0],torch.Tensor):
                    predictions = [x.detach().cpu().numpy() for x in predictions]
                    predictions = [x.astype(int) for x in predictions]
            elif predictions is None:
                predictions = [[x] for x in prediction]
            if len(prediction)!=len(predictions):
                raise ValueError("Size mismatch between prediction and prediction(s). {} != {}".format(
                    len(prediction),len(predictions)))
            if predictive_status is None:
                df_dict = dict({'prediction':prediction,
                           'classification':classification,
                           'predictions':predictions})
                data = pd.DataFrame.from_dict(df_dict, orient='columns')
                data['predictive_status']=data.apply(lambda x: 'TP' if x.classification in x.predictions else 'FP',axis=1)
            else:
                if predictions is not None:
                    if (len(predictions)!=len(predictive_status)):
                        raise ValueError("Size mismatch between predictions and predictive status. {} != {}".format(
                        len(predictions),
                        len(predictive_status)))
                else:
                    df_dict =  dict({'prediction':prediction,
                                     'classification':classification,
                                     'predictions':predictions,
                                     'predictive_status':predictive_status})
                data = pd.from_dict(df_dict, orient='index')
            return data
        except Exception as E:
            print("Raised the following error:{}".format(E))

    def apply_classification_scheme(self,classification_scheme:None,**params):
        """ Method for transforming the assignment dataframe with an external function and additional parameters
        """
        if classification_scheme:
            self.data = classification_scheme(self.data,**params)
    
    def to_dict(self):
        """ Convert instance attributes to a dictionary."""
        return self.__dict__
        
    def display(self):
        """ Display the Assignment_DF
        """
        print(self.data)
    
    def save(self,address:str,name:Optional[str]=None):
        """ Save a pickled representing the attributes of the Assignment DF object"""
        dict_to_save = self.to_dict()
        if name:
            assignment_df_name = name+".pickle"
        else:
            assignment_df_name = "_assignment_df.pickle"
        if pickle:
            with open(os.path.join(address,assignment_df_name),'wb') as handle:
                pickle.dump(dict_to_save, handle,protocol=pickle.HIGHEST_PROTOCOL)

    def load(self,address:str,name:Optional[str]=None):
        try:
            #open
            if name:
                assignment_df_name = name+".pickle"
            else:
                assignment_df_name = "_assignment_df.pickle"
            with open(os.path.join(address,assignment_df_name),'rb') as file:
                data = pickle.load(file) 
            for name,attr in data.items():
                setattr(self,name,attr)
        except Exception as E:
            print("Failed to load due to the following error:\n {}".format(E))
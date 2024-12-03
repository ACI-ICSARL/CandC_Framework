# certainty_distribution.py
import pandas as pd
import os
import pickle

from typing import Optional

class Certainty_Distribution():
    f""" The Certainty_Distribution object is a particular form of the pandas dataframe the collects the relevant information from certainties and inputs 
    """
    def __init__(self, data:Optional[pd.DataFrame]=None, address:Optional[str]=None,name:Optional[str]=None):
        try:
            if isinstance(data,pd.DataFrame):
                self.data = data
            elif address:
                self.data = self.load(address=address,name=name)
            else:
                raise ValueError("Missing data or address")
        except Exception as E:
            print("Raised the following error while initializing the Certainty Distribution \n {}".format(E)) 
            
    def display(self):
        """ Display the Certainty_Distribution.
        """
        print(self.data)

    def to_dict(self):
        """ Convert instance attributes to a dictionary."""
        return self.__dict__
        
    def save(self,address:str,name:Optional[str]=None):
        """ Save a pickled representing the attributes of the Certainties object"""
        dict_to_save = self.to_dict()
        if name:
            cert_dist_name = name+".pickle"
        else:
            cert_dist_name = "_cert_dist.pickle"
        if pickle:
            with open(os.path.join(address,cert_dist_name),'wb') as handle:
                pickle.dump(dict_to_save, handle,protocol=pickle.HIGHEST_PROTOCOL)

    def load(self,address:str,name:Optional[str]=None):
        """Load in the saved representation of the Certainty Distribution data.
        """
        if name:
            cert_dist_name = name+".pickle"
        else:
            cert_dist_name = "_cert_dist.pickle"
        with open(os.path.join(address,cert_dist_name), 'rb') as file:
            cert_dist = pickle.load(file)
        setattr(self,'data',cert_dist)
# gather methods
from typing import Union, Optional
from itertools import product
from tqdm import tqdm

from ..model_data import *
#########################################################
#
#   GATHERLIST methods
#
#########################################################
    
def make_full_gatherlist(model_data:Union[Input_Data,Output_Data,Model_Data]):
    """ Given input or output model_data with 'classification_categories' attribute, generate a full list of 
         categories by predictive status for subsequent processing.
    """
    classifications =['Global']+list(model_data['classification_categories'].keys())
    status=['','TP','FP']
    return  product(classifications,status)

def make_full_TP_gatherlist(model_data):
    """ Given input or output model_data with 'classification_categories' attribute, generate a list to gather
    the true positive data by category.
    """
    classifications =['Global']+list(model_data['classification_categories'].keys())
    status=['TP']
    return  product(classifications,status)    
    
def _df_condition_generator(classification:Union[int,str],predictive_status:str):
    """ Given classification status, either 'Global' or specific categoy """
    try:
        if classification=='Global':
            if predictive_status=='':
                return (lambda x: True)
            else:
                return (lambda x: x.predictive_status == predictive_status)
        else:
            if predictive_status=='':
                return (lambda x: x.classification == classification)
            else:
                return (lambda x: (x.classification == classification) & (x.predictive_status== predictive_status))
    except:
        print('Not given appropriate name structure; returning identity function')
        return (lambda x:True)
        
def _get_indices_by_criteria(criteria,assignment_df:pd.DataFrame):
    """ Returns the indices of the assignment data frame which satisfy criteria. Criteria in this case
        is a lambda expression containing the logical conditions to be satisfied, or otherwise a boolean
        condition applied to the assignmentdf data frame directly.
    """    
    tempdf=assignment_df.reset_index()
    tempdf['classification']=tempdf['classification'].astype(int)
    indices= tempdf.loc[tempdf.apply((criteria),axis=1)].index.tolist()
    return indices

#####################################################
#
# Gather for all methods
#
######################################################
def gather_for_all(func,assignment_df:pd.DataFrame,verbose=True,*gatherlist,**reference_frames):
    """ Given a function gathers statistics from an indexed reference frame, this function gathers 
    that statistic when conditioned on both global and categorical assignment, and further, by 
    Predictive Status, TP/FP, returning a dictionary of said statistics.
    """
    score_dict=dict()
    for ref_key,ref_frame in reference_frames.items():
        for name in tqdm(gatherlist):
            name_as_str=':'.join([str(elt) for elt in name])
            while name_as_str[-1]==":":
                name_as_str= name_as_str[0:-1]
            classification=name[0]
            predictive_status=name[1]
            if classification!='Global':
                classification =int(classification)
            indices = _get_indices_by_criteria(criteria=_df_condition_generator(classification,predictive_status),assignment_df=assignment_df)
            #print("the indices for classification {} and predictive status {} are {}".format(classification,predictive_status,indices))
            if verbose:
                print('Gathering  {} stats for {} from {}'.format(func.__name__,name_as_str, type(ref_frame)))
            if type(ref_frame) in [pd.DataFrame, pd.Series]:
                if indices!=[]:
                    score_dict.__setitem__(name_as_str, func(ref_frame.iloc[indices]))
                else:
                    score_dict.__setitem__(name_as_str, None)
            else:
                rrefframe = ref_frame[indices]
                if indices!=[]:
                    score_dict.__setitem__(name_as_str, func(rrefframe))
                else:
                    score_dict.__setitem__(name_as_str, None)
    return score_dict

def gather_for_all_pair_output(func,verbose=True,*gatherlist,**reference_frames):
    """ Given a function that gathers statistics from an indexed reference frame and returns a pair of dictionary objects,
        this function gathers that statistic when conditioned on both global and categorical assignment, and further, 
        by Predictive Status, TP/FP, returning a dictionary of said statistics.
    """
    score_dict_one=dict()
    score_dict_two=dict()
    for ref_key,ref_frame in reference_frames.items():
        for name in tqdm(gatherlist):
            name_as_str=':'.join([str(elt) for elt in name])
            while name_as_str[-1]==":":
                name_as_str= name_as_str[0:-1]
            classification=name[0]
            predictive_status=name[1]
            if classification!='Global':
                classification =int(classification)
            indices = _get_indices_by_criteria(criteria=_df_condition_generator(classification,predictive_status))
            if verbose:
                print('Gathering  {} stats for {} from {}'.format(func.__name__,name_as_str, type(ref_frame)))
            if type(ref_frame) in [pd.DataFrame, pd.Series]:
                if indices!=[]:
                    objone,objtwo =func(ref_frame.iloc[indices])
                    score_dict_one.__setitem__(name_as_str, objone)
                    score_dict_two.__setitem__(name_as_str,objtwo)
                else:
                    score_dict_one.__setitem__(name_as_str, None)
                    score_dict_two.__setitem__(name_as_str,None)
            else:
                rrefframe = ref_frame[indices]
                if indices!=[]:
                    objone,objtwo =func(rrefframe)
                    score_dict_one.__setitem__(name_as_str, objone)
                    score_dict_two.__setitem__(name_as_str,objtwo)
                else:
                    score_dict_one.__setitem__(name_as_str, None)
                    score_dict_two.__setitem__(name_as_str, None)
    return score_dict_one,score_dict_two
    
def gather_for_all_from_two_frames(func,assignment_df:pd.DataFrame,*gatherlist,**reference_frame_pairs):
    """Function for use to iterate through a gather list and reference frame for comparion between pairs. Outputs a single dictionary."""
    score_dict=dict()
    for _,ref_pairs in reference_frame_pairs.items():
        for name in tqdm(gatherlist):
            score_key =str(name[0])+':'+str(name[1])
            if score_key[-1]==':':
                score_key=score_key[0:-1]
            classification=name[0]
            predictive_status=name[1]
            if classification!='Global':
                classification =int(classification)
                indices = _get_indices_by_criteria(criteria=_df_condition_generator(classification,predictive_status),assignment_df=assignment_df)
            if (type(ref_pairs[0]) in [pd.DataFrame, pd.Series]) and type(ref_pairs[1]) in [pd.DataFrame, pd.Series]:
                if indices!=[]:
                    score_dict.__setitem__(score_key, func(ref_pairs[0].iloc[indices],ref_pairs[1].iloc[indices]))
                else:
                    score_dict.__setitem__(score_key, None)
            elif type(ref_pairs[1]) in [pd.DataFrame, pd.Series]:
                if indices!=[]:
                    score_dict.__setitem__(score_key, func(ref_pairs[0][indices],ref_pairs[1].iloc[indices]))
                else:
                    score_dict.__setitem__(score_key, None)
            elif(type(ref_pairs[0]) in [pd.DataFrame, pd.Series]) :
                if indices!=[]:
                    score_dict.__setitem__(score_key, func(ref_pairs[0].iloc[indices],ref_pairs[1][indices]))
                else:
                    score_dict.__setitem__(score_key, None)
            else:
                if indices!=[]:
                    score_dict.__setitem__(score_key, func(ref_pairs[0][indices],ref_pairs[1][indices]))
                else:
                    score_dict.__setitem__(score_key, None)
    return score_dict

def gather_stats_and_tests_for_all_from_two_frames(func,assignment_df:pd.DataFrame,*gatherlist,**reference_frame_pairs):
    """
    Function for use to iterate through a gather list and reference frame for comparion between pairs. 
    Outputs two dictionaries, one for the collective statistics, and the other for the colletive subtests run.
    Use if func returns a pair.
    """
    score_dict=dict()
    test_dict=dict()
    for _,ref_pairs in reference_frame_pairs.items():
        for name in tqdm(gatherlist):
            score_key =str(name[0])+':'+str(name[1])
            if score_key[-1]==':':
                score_key=score_key[0:-1]
            classification=name[0]
            predictive_status=name[1]
            if classification!='Global':
                classification =int(classification)
            indices = _get_indices_by_criteria(criteria=_df_condition_generator(classification,predictive_status),assignment_df=assignment_df)
            if (type(ref_pairs[0]) in [pd.DataFrame, pd.Series]) and type(ref_pairs[1]) in [pd.DataFrame, pd.Series]:
                if indices!=[]:
                    scores,tests=func(ref_pairs[0].iloc[indices],ref_pairs[1].iloc[indices])
                    score_dict.__setitem__(score_key,scores)
                    test_dict.__setitem__(score_key,tests)
                else:
                    score_dict.__setitem__(score_key, None)
                    test_dict.__setitem__(score_key, None)
            elif type(ref_pairs[1]) in [pd.DataFrame, pd.Series]:
                if indices!=[]:
                    scores,tests=func(ref_pairs[0][indices],ref_pairs[1].iloc[indices])
                    score_dict.__setitem__(score_key,scores)
                    test_dict.__setitem__(score_key,tests)
                else:
                    score_dict.__setitem__(score_key, None)
                    test_dict.__setitem__(score_key, None)
            elif(type(ref_pairs[0]) in [pd.DataFrame, pd.Series]) :
                if indices!=[]:
                    scores,tests=func(ref_pairs[0].iloc[indices],ref_pairs[1][indices])
                    score_dict.__setitem__(score_key,scores )
                    test_dict.__setitem__(score_key,tests )
                else:
                    score_dict.__setitem__(score_key, None)
                    test_dict.__setitem__(score_key, None)
            else:
                if indices!=[]:
                    scores,tests= func(ref_pairs[0][indices],ref_pairs[1][indices])
                    score_dict.__setitem__(score_key, scores)
                    test_dict.__setitem__(score_key,tests)
                else:
                    score_dict.__setitem__(score_key, None)
                    test_dict.__setitem__(score_key, None)
    return score_dict,test_dict
                

def gather_for_all_from_two_frames_with_companion_dict(func,assignment_df:pd.DataFrame,companion_dict=dict(),*gatherlist,**reference_frame_pairs):
    score_dict=dict()
    for _,ref_pairs in reference_frame_pairs.items():
        for name in tqdm(gatherlist):
            score_key =str(name[0])+':'+str(name[1])
            if score_key[-1]==':':
                score_key=score_key[0:-1]
            classification=name[0]
            predictive_status=name[1]
            if classification!='Global':
                classification =int(classification)
            indices = _get_indices_by_criteria(criteria=_df_condition_generator(classification,predictive_status),assignment_df=assignment_df)
            if (type(ref_pairs[0]) in [pd.DataFrame, pd.Series]) and type(ref_pairs[1]) in [pd.DataFrame, pd.Series]:
                if indices!=[]:
                    score_dict.__setitem__(score_key, func(ref_pairs[0].iloc[indices],ref_pairs[1].iloc[indices],**companion_dict))
                else:
                    score_dict.__setitem__(score_key, None)
            elif type(ref_pairs[1]) in [pd.DataFrame, pd.Series]:
                if indices!=[]:
                    score_dict.__setitem__(score_key, func(ref_pairs[0][indices],ref_pairs[1].iloc[indices],**companion_dict))
                else:
                    score_dict.__setitem__(score_key, None)
            elif(type(ref_pairs[0]) in [pd.DataFrame, pd.Series]) :
                if indices!=[]:
                    score_dict.__setitem__(score_key, func(ref_pairs[0].iloc[indices],ref_pairs[1][indices],**companion_dict))
                else:
                    score_dict.__setitem__(score_key, None)
            else:
                if indices!=[]:
                    score_dict.__setitem__(score_key, func(ref_pairs[0][indices],ref_pairs[1][indices],**companion_dict))
                else:
                    score_dict.__setitem__(score_key, None)
    return score_dict
    
def gather_for_all_with_predictive_comparison(func,assignment_df:pd.DataFrame,func_name='',verbose=True,*gatherlist,**reference_frames):
    """ Given a function gathers statistics from an indexed reference frame, this function gathers that statistic when conditioned on both
    global and categorical assignment, and further, by Predictive Status, TP/FP, returning a dictionary of said statistics.
    """
    score_dict=dict()
    for ref_key,ref_frame in reference_frames.items():
        for name in tqdm(gatherlist):
            classification=name
            if classification!='Global':
                classification =int(classification)
            fpindices = _get_indices_by_criteria(_df_condition_generator(classification,'FP'),assignment_df=assignment_df)
            tpindices = _get_indices_by_criteria(_df_condition_generator(classification,'TP'),assignment_df=assignment_df)
            if type(ref_frame) in [pd.DataFrame, pd.Series]:
                if tpindices!=[] and fpindices!=[]:
                    score_dict.__setitem__(str(name)+' '+func_name+' scores', func(ref_frame.iloc[tpindices],ref_frame.iloc[fpindices]))
                else:
                    score_dict.__setitem__(str(name)+' '+func_name+' scores',None)
            else:
                if tpindices!=[] and fpindices!=[]:
                    score_dict.__setitem__(str(name)+' '+func_name+' scores', func(ref_frame[tpindices],ref_frame[fpindices]))
                else:
                    score_dict.__setitem__(str(name)+' '+func_name+' scores',None)
    return score_dict

def gather_for_all_with_predictive_comparison_using_internal_test_dict(func,assignment_df:pd.DataFrame,test_dict_name,oodd_tests:dict,verbose=True,*gatherlist,**reference_frames):
    """Method for iterating through category labels with respect to lambda expressions (func) taking three arguments, 
    where the first two arguments correspond to the in category contrary predictive status objects being compared, 
    and the third argument corresponds to the oodd_tests being applied.
    """
    score_dict=dict()
    for ref_key,ref_frame in reference_frames.items():
        for name in tqdm(gatherlist):
            name_as_str=':'.join([str(elt) for elt in name])
            while name_as_str[-1]==":":
                name_as_str= name_as_str[0:-1]      
            classification=name
            if classification!='Global':
                classification =int(classification)
            fpindices = _get_indices_by_criteria(_df_condition_generator(classification,'FP'),assignment_df=assignment_df)
            tpindices = _get_indices_by_criteria(_df_condition_generator(classification,'TP'),assignment_df=assignment_df)
            if type(ref_frame) in [pd.DataFrame, pd.Series]:
                if tpindices!=[] and fpindices!=[]:
                    score_dict.__setitem__(str(name)+' '+func_name+' scores',
                                           func(ref_frame.iloc[tpindices],
                                                ref_frame.iloc[fpindices],
                                                oodd_tests[test_dict_name][name_as_str]))
                else:
                    score_dict.__setitem__(str(name)+' '+func_name+' scores',None)
            else:
                if tpindices!=[] and fpindices!=[]:
                    score_dict.__setitem__(str(name)+' '+func_name+' scores', func(ref_frame[tpindices],ref_frame[fpindices],oodd_tests[test_dict_name][name_as_str]))
                else:
                    score_dict.__setitem__(str(name)+' '+func_name+' scores',None)
    return score_dict

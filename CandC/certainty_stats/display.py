#display
import pandas as pd

import plotly.express as px


def make_label_table(df,values=['count','median'],**dist_stats):
    """ Returns pandas.DataFrames gathering statistics of certainty distributions (tabledf), a DataFrame for displaying competence scores (compdf), and a DataFrame for displaying the
    Mann Whitney U scores (mwdf).
    
    Parameters
    ----------
    df : pd.DataFrame
        input dataframe should contain certainty score data frame, with 
    values: list
        Default list consists of gathering 'count' and 'median' statistics, can include ['count', 'min','max','median','mean', 'std']
    dist_stats: dict()
        The dictionary of the corresponding input df's distribution statistics.
    
    Returns
    -------
    tabledf : pandas.Dataframe
        Collated dataframe of general statistical information targets by values argument
    compdf : pandas.Dataframe
        Collated competence score information for global and local levels
    mwdf : pandas.Dataframe
        Collated Mann Whitney U test statistics for global and local TP and FP levels.
    """
    columns_list = []
    for val in values:
        columns_list.append(('certainty_score',val))
    label_table = dict()
    comp_table = dict()
    mw_table = dict()
    for label in df.prediction.unique():
        label_table.__setitem__(label,dist_stats[label]['stats'][columns_list])
        comp_table.__setitem__(label,dist_stats[label]['empirical competence'])
        mw_table.__setitem__(label,dist_stats[label]['mann-whitney'])
    tabledf = pd.concat(label_table.values(),axis=1,keys=label_table.keys())
    compdf = pd.concat(comp_table.values(),axis=1,keys=comp_table.keys())
    mwdf = pd.concat(mw_table.values(),axis=1,keys=mw_table.keys())
    tabledf.columns = tabledf.columns.droplevel(1)
    return tabledf, compdf, mwdf
    
def cert_box_plot(cert_dist,title=""):
    """ Returns plotly.express box plots of distributions of TP and False positive certainty scores for an certainty_distribution (cert_dist)
    
    Parameters
    ----------
    cert_dist : pd.DataFrame
        Must be a pandas Dataframe containing the certainty score dataframe, e.g. Dataframe with information for columns {certainty_score, prediction, predictive_status}
    title: str
        Defaults to empty string, otherwise, user may enter title for plotly.express box plot figure.
    """
    fig = px.box(dataset,y="certainty_score",x="prediction",color="predictive_status",color_discrete_map={'test':'blue','TP':'green','FP':'red'},range_y=[-0.1,1.1],title = title,points=False)
    return fig

def make_dist_plots(**df_dict):
    """ Returns a dictionary consisting of plotly express box plots
    
    Parameters
    ----------
    df_dict : dict()
        Key,value pairs should be name of the corresponding subDataFrame of a certainty distribution, e.g. {'Label1': cert_dist[cert_dist.prediction=='Label1'],..} would be a valid key-value pair
        that makes the cert_box_plot for the sub dataframe consisting of all samples that were predicted to be Label1, including the TP, FP, and test samples
        
    Returns
    -------
    plot_dict: dict()
        A dictionary object consisting of the plotly.express box plots
    """
    plot_dict = dict()
    for name,cert_df in df_dict.items():
        plot_dict.__setitem__(name, cert_box_plot(cert_dist=cert_df,title=name))
    return plot_dict



def make_violin_fig(control_df,test_df,testname,**params):
    """ Returns a plotly graph object violin figure
    
    Parameters
    ----------
    control_df: pandas.Dataframe
        reference dataframe containing as a subdataframe the reference certainty distribution
    test_df: pandas.Dataframe
        dataframe of test data, containing as a subdataframe the certainty distribution of the test data
    test_name: str
        string corresponds to optional parameters used in the out_of_distribution_detection function that are passed to make_violin_figs
    params: dict()
        optional parameters intended to configure the violin figure generated further
    
    Returns
    -------
    fig: plotly.graph_object
    """
    fig = go.Figure()
    test_params = dict()
    test_params.__setitem__('x',test_df.prediction)
    test_params.__setitem__('y',test_df.certainty_score)
    test_params.__setitem__('line_color','blue')
    for key in params[testname]['plot_params'].keys():
        test_params.__setitem__(key,test_params[testname]['plot_params'][key])
        fig.add_trace(go.Violin(**test_params))
        fig.add_trace(go.Violin(x= control_df.prediction[control_df.predictive_status=='TP'],y=control_df.certainty_score[control_df.predictive_status=='TP'], line_color='green',legendgroup='TP',scalegroup='TP',name='TP',points=False))
        fig.add_trace(go.Violin(x= control_df.prediction[control_df.predictive_status=='FP'],y=control_df.certainty_score[control_df.predictive_status=='FP'], line_color='red',legendgroup='FP',scalegroup='FP',name='FP',points=False))
        fig.update_traces(box_visible=True,meanline_visible=True)
        fig.update_layout(violinmode='group')
        fig.update_layout(legend=dict(orientation='h', yanchor='top',y=1.15, xanchor='left', x=0),yaxis=dict(title_text='Certainty'),xaxis=dict(title_text='Prediction'),)
    return fig
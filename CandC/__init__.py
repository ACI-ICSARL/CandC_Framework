# __init__.py
"""
CandC Framework
********************************************
A module for the computation and evaluation of model certainty and competence, with additional tools for out-of-distribution detection (OODD) and evaluating model uncertainty with the tools of the CandC framework.
"""
__version__="0.0.2"

from . import candc, certainty_stats, loss, oodd, utils, model_uq
from numpy import array, nan

__all__=["candc","certainty_stats","loss","oodd","utils","model_uq","__version__"]


##############
#from .certainty import component_certainty,complete_component_certainty, get_certainty, get_upper_cert_as_vec, get_batch_certainties,get_batch_upper_cert_as_vec
#from .certainty_stats import  predictive_entropy, mutual_information_prediction, vr, vro, , omicron_fn, omicron_stats, find_cert_dist_dataframe, dist_stats, make_label_table, cert_box_plot, make_dist_plots, make_violin_fig
#from .oodd_tests import  out_of_distribution_detection, omicron_test, omicron_ecdf_test,mwu_cert_dist_test,  omicron_test_results, pcs_vro_test, mwu_cert_dist_test_tranches
#from numpy import array, nan
#from .cost import chi, vartheta, cost_fn
#from .model_UQ import model_UQ

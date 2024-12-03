"""
gathe:
*******************************************************
This subfolder contains the various gather functions 

"""
__version__="0.0.1"

from .gather_methods import (
    make_full_gatherlist,
    make_full_TP_gatherlist,
    gather_for_all, 
    gather_for_all_pair_output,
    gather_for_all_from_two_frames,
    gather_stats_and_tests_for_all_from_two_frames,
    gather_for_all_from_two_frames_with_companion_dict,
    gather_for_all_with_predictive_comparison,
    gather_for_all_with_predictive_comparison_using_internal_test_dict,
)
__all__ = ['make_full_gatherlist','make_full_TP_gatherlist','gather_for_all','gather_for_all_pair_output',    'gather_for_all_from_two_frames','gather_stats_and_tests_for_all_from_two_frames', 'gather_for_all_from_two_frames_with_companion_dict','gather_for_all_with_predictive_comparison','gather_for_all_with_predictive_comparison_using_internal_test_dict', '__version__']
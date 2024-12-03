"""
Certainty and Competence (CandC)

This module provides the core components used in the certainty and competence framework, namely, the tools needed to compute certainty and from certainty, the tools used to compute competence.
"""

from .certainty import component_certainty,complete_component_certainty, get_certainty, get_upper_cert_as_vec, get_batch_certainties,get_batch_upper_cert_as_vec
from .competence import component_competence, empirical_competence
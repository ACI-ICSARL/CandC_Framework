# __init__.py
"""
MODEL UQ
********************************************
The submodule containing the interface for the model uncertainty quantification, out-of-distribution detection (OODD) and evaluation of model uncertainty with the tools of the CandC framework.
"""
__version__="0.0.1"

from . import data
from .base import Model_UQ
__all__=["__version__"]


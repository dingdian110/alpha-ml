import os
from alphaml.engine.components.models.base_model import BaseRegressionModel
from alphaml.utils.class_loader import find_components

"""
Load the buildin regressors.
"""
regressors_directory = os.path.split(__file__)[0]
_regressors = find_components(__package__, regressors_directory, BaseRegressionModel)

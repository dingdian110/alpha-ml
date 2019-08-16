import os
from alphaml.engine.components.models.base_model import BaseClassificationModel
from alphaml.utils.class_loader import find_components

"""
Load the buildin classifiers.
"""
hyperopt_classifiers_directory = os.path.split(__file__)[0]
_hyperopt_classifiers = find_components(__package__, hyperopt_classifiers_directory, BaseClassificationModel)

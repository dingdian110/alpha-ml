import os
from alphaml.engine.components.models.base_model import BaseImageClassificationModel
from alphaml.utils.class_loader import find_components

"""
Load the buildin classifiers.
"""
classifiers_directory = os.path.split(__file__)[0]
_img_classifiers = find_components(__package__, classifiers_directory, BaseImageClassificationModel)
print(_img_classifiers)
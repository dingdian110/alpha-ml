from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import keras
from keras_applications import get_keras_submodule
from keras import Model
from keras.layers import Dropout, Dense
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    InCondition
from alphaml.engine.components.models.base_dl_model import BaseImageClassificationModel
from alphaml.engine.components.data_preprocessing.image_preprocess import preprocess
from alphaml.utils.constants import *

backend = get_keras_submodule('backend')
engine = get_keras_submodule('engine')
layers = get_keras_submodule('layers')
models = get_keras_submodule('models')
keras_utils = get_keras_submodule('utils')
keras_utils = get_keras_submodule('utils')
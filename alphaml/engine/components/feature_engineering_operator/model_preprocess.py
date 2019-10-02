
from alphaml.engine.components.feature_engineering_operator.encoder import *


def preprocess_xgboost(dm):
    dm = one_hot(dm)
    return dm


def preprocess_lightgbm(dm):
    return dm


def preprocess_randomforest(dm):
    return dm


def preprocess_adaboost(dm):
    return dm


def preprocess_fm(dm):
    dm = one_hot(dm)
    return dm
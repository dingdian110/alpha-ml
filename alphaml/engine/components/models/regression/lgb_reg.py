import xgboost as xgb

from alphaml.engine.components.models.base_model import BaseClassificationModel
from alphaml.utils.constants import *


from xgboost.sklearn import XGBClassifier


class LightGBMRegressor():

    def __init__(self, max_depth, eta, num_round):
        # super(Xgboost, self).__init__()
        self.max_depth = max_depth
        self.eta = eta
        self.num_round = num_round
        self.estimator = None

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, y)
        params = {"max_depth": self.max_depth,
                  "eta": self.eta,
                  "objective": ""
                  }
        self.estimator = xgb.train(params, dtrain, self.num_round)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError("The model has not been fitted")
        dtest = xgb.DMatrix(X)
        return self.estimator.predict(dtest)

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError("The model has not been fitted")
        dtest = xgb.DMatrix(X)
        return self.estimator.predict(dtest)

    @staticmethod
    def get_properties():
        return {'shortname': 'LGBR',
                'name': 'LightGBM Regressor',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    def get_estimator(self):
        return self.estimator

    @staticmethod
    def get_hyperparameter_search_space():
        pass

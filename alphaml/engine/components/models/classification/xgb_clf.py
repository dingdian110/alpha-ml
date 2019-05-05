import xgboost as xgb

from alphaml.engine.components.models.base_model import BaseClassificationModel
from alphaml.utils.constants import *

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter


from xgboost.sklearn import XGBClassifier


class XgboostClassifier(BaseClassificationModel):

    def __init__(self, max_depth, learning_rate, n_estimators, gamma):
        # super(Xgboost, self).__init__()
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.gamma = gamma

        self.estimator = None

    def iterative_fit(self, X, y, sample_weight=None, n_iter=1, refit=False):
        pass

    def fit(self, X, y):
        n_class = len(set(y))
        objective = str()
        if n_class == 2:
            objective = "binary:logistic"
        elif n_class > 2:
            objective = "multi:softmax"
        else:
            raise RuntimeError('the number of class if less than 2')

        self.estimator = XGBClassifier(max_depth=self.max_depth,
                                       learning_rate=self.learning_rate,
                                       n_estimators=self.n_estimators,
                                       gamma=self.gamma,
                                       silent=True,
                                       objective=objective
                                       )
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError("The model has not been fitted")
        return self.estimator.predict_proba(X)

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError("The model has not been fitted")
        return self.estimator.predict(X)

    @staticmethod
    def get_properties():
        return {'shortname': 'XGBC',
                'name': 'Xgboost Classifier',
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
        max_depth = UniformIntegerHyperparameter("max_depth", 1, 10, default_value=6)
        learning_rate = UniformFloatHyperparameter("learning_rate", 1e-5, 1.0, default_value=0.3)
        n_estimators = UniformIntegerHyperparameter("n_estimators", 50, 500, default_value=100)
        gamma = UniformFloatHyperparameter("gamma", 0, 100, default_value=1.0)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([max_depth, learning_rate, n_estimators, gamma])

        return cs

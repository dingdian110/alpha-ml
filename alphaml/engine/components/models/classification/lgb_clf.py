from alphaml.engine.components.models.base_model import BaseClassificationModel
from alphaml.utils.constants import *

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter

from lightgbm.sklearn import LGBMClassifier


class LightGBM(BaseClassificationModel):

    def __init__(self, num_leaves=31, max_depth=-1,
                 learning_rate=0.1, n_estimators=100):
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators

        self.estimator = None

    def fit(self, X, y):
        self.estimator = LGBMClassifier(num_leaves=self.num_leaves,
                                        max_depth=self.max_depth,
                                        learning_rate=self.learning_rate,
                                        n_estimators=self.n_estimators)
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
        return {'shortname': 'LGBC',
                'name': 'LightGBM Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_estimator(self):
        return self.estimator

    @staticmethod
    def get_hyperparameter_search_space():
        num_leaves = UniformIntegerHyperparameter("num_leaves", 5, 50, default_value=31)
        max_depth = UniformIntegerHyperparameter("max_depth", 3, 10, default_value=5)
        learning_rate = UniformFloatHyperparameter("learning_rate", 1e-5, 1.0, default_value=0.1)
        n_estimators = UniformIntegerHyperparameter("n_estimators", 50, 500, default_value=100)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([num_leaves, max_depth, learning_rate, n_estimators])

        return cs

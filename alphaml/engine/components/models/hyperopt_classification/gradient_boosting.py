import numpy as np
import sklearn.ensemble
from hyperopt import hp

from alphaml.engine.components.models.base_model import BaseClassificationModel, IterativeComponentWithSampleWeight
from alphaml.utils.common import check_none
from alphaml.utils.constants import *


class GradientBoostingClassifier(IterativeComponentWithSampleWeight, BaseClassificationModel):
    def __init__(self, loss, learning_rate, n_estimators, subsample,
                 min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, max_depth, criterion, max_features,
                 max_leaf_nodes, min_impurity_decrease, random_state=None,
                 verbose=0):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.verbose = verbose
        self.estimator = None
        self.fully_fit_ = False

    def iterative_fit(self, X, y, sample_weight=None, n_iter=1, refit=False):

        # Special fix for gradient boosting!
        if isinstance(X, np.ndarray):
            X = np.ascontiguousarray(X, dtype=X.dtype)
        if refit:
            self.estimator = None

        if self.estimator is None:
            self.learning_rate = float(self.learning_rate)
            self.n_estimators = int(self.n_estimators)
            self.subsample = float(self.subsample)
            self.min_samples_split = int(self.min_samples_split)
            self.min_samples_leaf = int(self.min_samples_leaf)
            self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)
            if check_none(self.max_depth):
                self.max_depth = None
            else:
                self.max_depth = int(self.max_depth)
            self.max_features = float(self.max_features)
            if check_none(self.max_leaf_nodes):
                self.max_leaf_nodes = None
            else:
                self.max_leaf_nodes = int(self.max_leaf_nodes)
            self.min_impurity_decrease = float(self.min_impurity_decrease)
            self.verbose = int(self.verbose)

            self.estimator = sklearn.ensemble.GradientBoostingClassifier(
                loss=self.loss,
                learning_rate=self.learning_rate,
                n_estimators=n_iter,
                subsample=self.subsample,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_depth=self.max_depth,
                criterion=self.criterion,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=True,
            )

        else:
            self.estimator.n_estimators += n_iter
            self.estimator.n_estimators = min(self.estimator.n_estimators,
                                              self.n_estimators)

        self.estimator.fit(X, y, sample_weight=sample_weight)

        # Apparently this if is necessary
        if self.estimator.n_estimators >= self.n_estimators:
            self.fully_fit_ = True

        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        return not len(self.estimator.estimators_) < self.n_estimators

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'GB',
                'name': 'Gradient Boosting Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        space = {'loss': hp.choice('gb_loss', ["deviance"]),
                 'learning_rate': hp.loguniform('gb_learning_rate', np.log(0.01), np.log(1)),
                 'n_estimators': hp.randint('gb_n_estimators', 451) + 50,
                 'max_depth': hp.randint('gb_max_depth', 10) + 1,
                 'criterion': hp.choice('gb_criterion', ['friedman_mse', 'mse', 'mae']),
                 'min_samples_split': hp.randint('gb_min_samples_split', 19) + 2,
                 'min_samples_leaf': hp.randint('gb_min_samples_leaf', 20) + 1,
                 'min_weight_fraction_leaf': hp.choice('gb_min_weight_fraction_leaf', [0]),
                 'subsample': hp.uniform('gb_subsample', 0.1, 1),
                 'max_features': hp.uniform('gb_max_features', 0.1, 1),
                 'max_leaf_nodes': hp.choice('gb_max_leaf_nodes', [None]),
                 'min_impurity_decrease': hp.choice('gb_min_impurity_decrease', [0])}

        init_trial = {'loss': "deviance", 'learning_rate': 0.1, 'n_estimators': 200, 'max_depth': 3,
                      'criterion': "friedman_mse", 'min_samples_split': 2, 'min_samples_leaf': 1,
                      'min_weight_fraction_leaf': 0, 'subsample': 1, 'max_features': 1,
                      'max_leaf_nodes': None, 'min_impurity_decrease': 0}
        return space

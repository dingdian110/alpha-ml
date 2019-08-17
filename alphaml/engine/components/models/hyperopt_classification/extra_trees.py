from hyperopt import hp

from alphaml.engine.components.models.base_model import BaseClassificationModel, IterativeComponentWithSampleWeight
from alphaml.utils.model_util import convert_multioutput_multiclass_to_multilabel
from alphaml.utils.common import check_none, check_for_bool
from alphaml.utils.constants import *


class ExtraTreesClassifier(IterativeComponentWithSampleWeight, BaseClassificationModel):

    def __init__(self, n_estimators, criterion, min_samples_leaf,
                 min_samples_split, max_features, bootstrap, random_state=None):

        if check_none(n_estimators):
            self.n_estimators = None
        else:
            self.n_estimators = int(self.n_estimators)
        self.criterion = criterion

        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = -1
        self.random_state = random_state

        self.estimator = None

    def fit(self, X, y, sample_weight=None):
        from sklearn.ensemble import ExtraTreesClassifier
        self.bootstrap = check_for_bool(self.bootstrap)
        self.estimator = ExtraTreesClassifier(n_estimators=self.n_estimators,
                                              max_leaf_nodes=None,
                                              criterion=self.criterion,
                                              max_features=self.max_features,
                                              min_samples_split=self.min_samples_split,
                                              min_samples_leaf=self.min_samples_leaf,
                                              max_depth=None,
                                              bootstrap=self.bootstrap,
                                              random_state=self.random_state,
                                              n_jobs=self.n_jobs)
        self.estimator.fit(X, y, sample_weight=sample_weight)
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
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'ET',
                'name': 'Extra Trees Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        space = {'n_estimators': hp.choice('et_n_estimators', [100]),
                 'criterion': hp.choice('et_criterion', ["gini", "entropy"]),
                 'max_features': hp.uniform('et_max_features', 0, 1),
                 'min_samples_split': hp.randint('et_min_samples_split', 19) + 2,
                 'min_samples_leaf': hp.randint('et_min_samples_leaf,', 20) + 1,
                 'bootstrap': hp.choice('et_bootstrap', ["True", "False"])}

        init_trial = {'n_estimators': 100, 'criterion': "gini", 'max_features': 0.5,
                      'min_samples_split': 2, 'min_samples_leaf': 1, 'bootstrap': "False"}
        return space

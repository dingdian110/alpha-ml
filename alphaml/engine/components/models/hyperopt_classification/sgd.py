import numpy as np
from hyperopt import hp

from alphaml.engine.components.models.base_model import BaseClassificationModel, IterativeComponentWithSampleWeight
from alphaml.utils.constants import *
from alphaml.utils.common import check_none, check_for_bool
from alphaml.utils.model_util import softmax


class SGD(
    IterativeComponentWithSampleWeight,
    BaseClassificationModel,
):
    def __init__(self, loss, penalty, alpha, fit_intercept, tol,
                 learning_rate, l1_ratio=0.15, epsilon=0.1,
                 eta0=0.01, power_t=0.5, average=False, random_state=None):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.learning_rate = learning_rate
        self.l1_ratio = l1_ratio
        self.epsilon = epsilon
        self.eta0 = eta0
        self.power_t = power_t
        self.random_state = random_state
        self.average = average
        self.estimator = None
        self.time_limit = None

    def iterative_fit(self, X, y, n_iter=2, refit=False, sample_weight=None):
        from sklearn.linear_model.stochastic_gradient import SGDClassifier

        # Need to fit at least two iterations, otherwise early stopping will not
        # work because we cannot determine whether the algorithm actually
        # converged. The only way of finding this out is if the sgd spends less
        # iterations than max_iter. If max_iter == 1, it has to spend at least
        # one iteration and will always spend at least one iteration, so we
        # cannot know about convergence.

        if refit:
            self.estimator = None

        if self.estimator is None:
            nested_loss = self.loss
            self.loss = nested_loss[0]
            if self.loss == 'modified_huber':
                self.epsilon = nested_loss[1]['epsilon']

            nested_penalty = self.penalty
            self.penalty = nested_penalty[0]
            if self.penalty == "elasticnet":
                self.l1_ratio = nested_penalty[1]['l1_ratio']

            nested_learning_rate = self.learning_rate
            self.learning_rate = nested_learning_rate[0]
            if self.learning_rate == 'invscaling':
                self.eta0 = nested_learning_rate[1]['eta0']
                self.power_t = nested_learning_rate[1]['power_t']
            elif self.learning_rate == 'constant':
                self.eta0 = nested_learning_rate[1]['eta0']

            self.fully_fit_ = False

            self.alpha = float(self.alpha)
            self.l1_ratio = float(self.l1_ratio) if self.l1_ratio is not None \
                else 0.15
            self.epsilon = float(self.epsilon) if self.epsilon is not None \
                else 0.1
            self.eta0 = float(self.eta0) if self.eta0 is not None else 0.01
            self.power_t = float(self.power_t) if self.power_t is not None \
                else 0.5
            self.average = check_for_bool(self.average)
            self.fit_intercept = check_for_bool(self.fit_intercept)
            self.tol = float(self.tol)

            self.estimator = SGDClassifier(loss=self.loss,
                                           penalty=self.penalty,
                                           alpha=self.alpha,
                                           fit_intercept=self.fit_intercept,
                                           max_iter=n_iter,
                                           tol=self.tol,
                                           learning_rate=self.learning_rate,
                                           l1_ratio=self.l1_ratio,
                                           epsilon=self.epsilon,
                                           eta0=self.eta0,
                                           power_t=self.power_t,
                                           shuffle=True,
                                           average=self.average,
                                           random_state=self.random_state,
                                           warm_start=True)
            self.estimator.fit(X, y, sample_weight=sample_weight)
        else:
            self.estimator.max_iter += n_iter
            self.estimator.max_iter = min(self.estimator.max_iter, 512)
            self.estimator._validate_params()
            self.estimator._partial_fit(
                X, y,
                alpha=self.estimator.alpha,
                C=1.0,
                loss=self.estimator.loss,
                learning_rate=self.estimator.learning_rate,
                max_iter=n_iter,
                sample_weight=sample_weight,
                classes=None,
                coef_init=None,
                intercept_init=None
            )

        if self.estimator.max_iter >= 512 or n_iter > self.estimator.n_iter_:
            self.fully_fit_ = True

        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        elif not hasattr(self, 'fully_fit_'):
            return False
        else:
            return self.fully_fit_

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()

        if self.loss in ["log", "modified_huber"]:
            return self.estimator.predict_proba(X)
        else:
            df = self.estimator.decision_function(X)
            return softmax(df)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'SGD Classifier',
                'name': 'Stochastic Gradient Descent Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        eta0 = hp.loguniform('sgd_eta0', np.log(1e-7), np.log(1e-1))
        space = {
            'loss': hp.choice('sgd_loss', [
                ("modified_huber", {'epsilon': hp.loguniform('sgd_epsilon', np.log(1e-5), np.log(1e-1))}),
                ("hinge", {}),
                ("log", {}),
                ("squared_hinge", {}),
                ("perceptron", {})]),
            'penalty': hp.choice('sgd_penalty',
                                 [("elasticnet", {'l1_ratio': hp.loguniform('sgd_l1_ratio', np.log(1e-9), np.log(1))}),
                                  ("l1", {}),
                                  ("l2", {})]),
            'alpha': hp.loguniform('sgd_alpha', np.log(1e-7), np.log(1e-1)),
            'fit_intercept': hp.choice('sgd_fit_intercept', ["True"]),
            'tol': hp.loguniform('sgd_tol', np.log(1e-5), np.log(1e-1)),
            'learning_rate': hp.choice('sgd_learning_rate', [("optimal", {}),
                                                             ("invscaling",
                                                              {'power_t': hp.loguniform('sgd_power_t', np.log(1e-5),
                                                                                        np.log(1)),
                                                               'eta0': eta0}),
                                                             ("constant", {'eta0': eta0})]),

            'average': hp.choice('sgd_average', ["True", "False"])}

        init_trial = {'loss': ("log", {}),
                      'penalty': ("l2", {}),
                      'alpha': 1e-4,
                      'fit_intercept': "True",
                      'tol': 1e-4,
                      'learning_rate': ("invscaling", {'power_t': 0.5, 'eta0': 0.01}),
                      'average': "False"}

        return space

import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter, \
    UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition

from alphaml.engine.components.models.base_model import BaseClassificationModel, IterativeComponentWithSampleWeight
from alphaml.utils.constants import *


class MLP(
    IterativeComponentWithSampleWeight,
    BaseClassificationModel,
):
    def __init__(self, activation, solver, alpha, tol, learning_rate='constant', learning_rate_init=0.001,
                 power_t=0.5, momentum=0.9, nesterovs_momentum=True,
                 beta1=0.9, random_state=None):
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.tol = tol
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.beta1 = beta1
        self.power_t = power_t
        self.random_state = random_state
        self.estimator = None
        self.fully_fit_ = False

    def iterative_fit(self, X, y, n_iter=2, refit=False, sample_weight=None):
        from sklearn.neural_network import MLPClassifier

        # Need to fit at least two iterations, otherwise early stopping will not
        # work because we cannot determine whether the algorithm actually
        # converged. The only way of finding this out is if the sgd spends less
        # iterations than max_iter. If max_iter == 1, it has to spend at least
        # one iteration and will always spend at least one iteration, so we
        # cannot know about convergence.

        if refit:
            self.estimator = None

        if self.estimator is None:
            self.fully_fit_ = False
            if self.learning_rate is None:
                self.learning_rate="constant"
            self.alpha = float(self.alpha)
            self.power_t = float(self.power_t) if self.power_t is not None \
                else 0.5
            self.tol = float(self.tol)

            self.estimator = MLPClassifier(activation=self.activation,
                                           solver=self.solver,
                                           alpha=self.alpha,
                                           learning_rate=self.learning_rate,
                                           learning_rate_init=self.learning_rate_init,
                                           power_t=self.power_t,
                                           max_iter=n_iter,
                                           shuffle=True,
                                           tol=self.tol,
                                           warm_start=True,
                                           momentum=self.momentum,
                                           nesterovs_momentum=self.nesterovs_momentum,
                                           beta_1=self.beta1)

        else:
            self.estimator.max_iter += n_iter
            self.estimator.max_iter = min(self.estimator.max_iter, 4096)
        self.estimator.fit(X, y)

        if self.estimator.max_iter >= 4096 or n_iter > self.estimator.n_iter_:
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
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'MLP Classifier',
                'name': 'Multi-layer Perceptron Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        activation = CategoricalHyperparameter("activation", ["identity", "logistic", "tanh", "relu"],
                                               default_value="relu")
        solver = CategoricalHyperparameter("solver", ["sgd", "adam"], default_value="adam")

        alpha = UniformFloatHyperparameter(
            "alpha", 1e-7, 1., log=True, default_value=0.0001)

        learning_rate = CategoricalHyperparameter(
            "learning_rate", ["adaptive", "invscaling", "constant"],
            default_value="constant")

        learning_rate_init = UniformFloatHyperparameter(
            "learning_rate_init", 1e-4, 3e-1, default_value=0.001, log=True)

        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-2, log=True,
                                         default_value=1e-4)
        momentum = UniformFloatHyperparameter("momentum", 0.6, 1, q=0.05, default_value=0.9)

        nesterovs_momentum = CategoricalHyperparameter("nesterovs_momentum", [True, False], default_value=True)
        beta1 = UniformFloatHyperparameter("beta1", 0.6, 1, default_value=0.9)
        power_t = UniformFloatHyperparameter("power_t", 1e-5, 1, log=True,
                                             default_value=0.5)
        cs.add_hyperparameters(
            [activation, solver, alpha, learning_rate, learning_rate_init, tol, momentum, nesterovs_momentum, beta1,
             power_t])

        # TODO add passive/aggressive here, although not properly documented?
        learning_rate_condition = EqualsCondition(learning_rate, solver, "sgd")
        momentum_condition = EqualsCondition(momentum, solver, "sgd")
        nesterovs_momentum_condition = EqualsCondition(nesterovs_momentum, solver, "sgd")
        beta1_condition = EqualsCondition(beta1, solver, "adam")

        power_t_condition = EqualsCondition(power_t, learning_rate,
                                            "invscaling")

        cs.add_conditions([learning_rate_condition, momentum_condition,
                           nesterovs_momentum_condition, beta1_condition, power_t_condition])

        return cs

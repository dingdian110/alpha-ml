from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition, InCondition, NotEqualsCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from alphaml.utils.constants import *
from alphaml.utils.model_util import softmax
from alphaml.engine.components.models.base_model import BaseClassificationModel


class Logistic_Regression(BaseClassificationModel):
    def __init__(self, C, penalty, solver, tol, max_iter, random_state=None):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.penalty = penalty
        self.solver = solver
        self.estimator = None

    def fit(self, X, Y):
        from sklearn.linear_model import LogisticRegression

        self.C = float(self.C)

        self.estimator = LogisticRegression(random_state=self.random_state,
                                            solver=self.solver,
                                            penalty=self.penalty,
                                            multi_class='multinomial')
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        decision = self.estimator.decision_function(X)
        return softmax(decision)


    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Logistic-Regression',
            'name': 'Logistic Regression Classification',
            'handles_regression': False,
            'handles_classification': True,
            'handles_multiclass': True,
            'handles_multilabel': False,
            'is_deterministic': True,
            'input': (DENSE, SPARSE, UNSIGNED_DATA),
            'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True,
                                       default_value=1.0)
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-3,
                                         log=True)

        max_iter = UniformIntegerHyperparameter("max_iter", 50, 1000, default_value=100)
        penalty = CategoricalHyperparameter(name="penalty",
                                           choices=["l1", "l2"],
                                           default_value="l2")
        solver = CategoricalHyperparameter(name="solver",
                                            choices=["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                                            default_value="liblinear")

        cs = ConfigurationSpace()
        cs.add_hyperparameters([C, penalty, solver, tol, max_iter])
        return cs

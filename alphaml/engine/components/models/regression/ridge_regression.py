from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from alphaml.utils.constants import *
from alphaml.engine.components.models.base_model import BaseRegressionModel


class Ridge_Regression(BaseRegressionModel):
    def __init__(self, alpha, solver, tol, max_iter, random_state=None):
        self.alpha = alpha
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from sklearn.linear_model import Ridge
        self.estimator = Ridge(alpha=self.alpha,
                               tol=self.tol,
                               max_iter=self.max_iter,
                               solver=self.solver,
                               random_state=self.random_state)
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Ridge-Regression',
                'name': 'Ridge Regression',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    def get_hyperparameter_search_space(dataset_properties=None):
        alpha = UniformFloatHyperparameter("alpha", 0.1, 32, log=True, default_value=1.0)
        tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2, default_value=1e-4,
                                         log=True)

        max_iter = UniformFloatHyperparameter("max_iter", 100, 1000, q=100, default_value=100)
        solver = CategoricalHyperparameter("solver", choices=["auto", "saga"], default_value="auto")

        cs = ConfigurationSpace()
        cs.add_hyperparameters([alpha, tol, max_iter, solver])
        return cs

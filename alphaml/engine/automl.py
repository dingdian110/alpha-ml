import numpy as np
from alphaml.engine.components.componets_manager import ComponentsManager
from alphaml.engine.optimizer.smac_smbo import SMAC_SMBO


class AutoML(object):
    def __init__(
            self,
            time_budget,
            each_run_budget,
            memory_limit,
            ensemble_size,
            include_models,
            exclude_models):
        self.time_budget = time_budget
        self.each_run_budget = each_run_budget
        self.ensemble_size = ensemble_size
        self.memory_limit = memory_limit
        self.include_models = include_models
        self.exclude_models = exclude_models
        self.component_manager = ComponentsManager()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        1. Define the ML pipeline.
           1) the configuration space.

        2. Search the promising configurations for this pipeline.

        :param X: array-like or sparse matrix of shape = [n_samples, n_features], the training input samples.
        :param y: array-like, shape = [n_samples], the training target.
        :return: self
        """
        # Detect the task type.
        task_type = 3
        # Get the configuration space for the automl task.
        config_space = self.component_manager.get_hyperparameter_search_space(
            task_type, self.include_models, self.exclude_models)

        smac_smbo = SMAC_SMBO(config_space)
        smac_smbo.run()
        return self

    def predict(self, X):
        return None

    def score(self, X, y):
        return None


class AutoMLClassifier(AutoML):
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        return super().fit(X, y)


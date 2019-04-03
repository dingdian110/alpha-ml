from alphaml.engine.components.componets_manager import ComponentsManager
from alphaml.engine.components.data_manager import DataManager
from alphaml.engine.evaluator.base import BaseEvaluator
from alphaml.engine.optimizer.smac_smbo import SMAC_SMBO
from alphaml.engine.optimizer.ts_smbo import TS_SMBO


class AutoML(object):
    def __init__(
            self,
            time_budget,
            each_run_budget,
            memory_limit,
            ensemble_size,
            include_models,
            exclude_models,
            optimizer='ts_smac'):
        self.time_budget = time_budget
        self.each_run_budget = each_run_budget
        self.ensemble_size = ensemble_size
        self.memory_limit = memory_limit
        self.include_models = include_models
        self.exclude_models = exclude_models
        self.component_manager = ComponentsManager()
        self.optimizer = optimizer

    def fit(self, data: DataManager, **kwargs):
        """
        1. Define the ML pipeline.
           1) the configuration space.

        2. Search the promising configurations for this pipeline.

        :param X: array-like or sparse matrix of shape = [n_samples, n_features], the training input samples.
        :param y: array-like, shape = [n_samples], the training target.
        :return: self
        """

        task_type = kwargs['task_type']
        metric = kwargs['metric']

        # Get the configuration space for the automl task.
        config_space = self.component_manager.get_hyperparameter_search_space(
            task_type, self.include_models, self.exclude_models)

        # Create evaluator & assign the required data to it.
        evaluator = BaseEvaluator(data, metric)

        if self.optimizer == 'smac':
            # Create optimizer.
            smac_smbo = SMAC_SMBO(config_space, evaluator)
            runhistory, _ = smac_smbo.run()

            # Show the results.
            configs = runhistory.get_all_configs()
            perfs = list()
            for config in configs:
                perfs.append(runhistory.get_cost(config))
            print(len(perfs))
            print(perfs)
        elif self.optimizer == 'ts_smac':
            # Create optimizer.
            ts_smbo = TS_SMBO(config_space, data, metric)
            ts_smbo.run()

        return self

    def predict(self, X):
        return None

    def score(self, X, y):
        return None


class AutoMLClassifier(AutoML):
    def fit(self, data, **kwargs):
        return super().fit(data, **kwargs)

import pandas as pd
from alphaml.engine.components.data_manager import DataManager


class BaseEstimator(object):
    """Base class for all estimators in alpha-ml. """

    def __init__(
            self,
            optimizer='mono_smbo',
            time_budget=3600,
            each_run_budget=360,
            ensemble_method='none',
            ensemble_size=10,
            memory_limit=1024,
            seed=None,
            include_models=None,
            exclude_models=None,
            tmp_dir=None,
            output_dir=None):
        self.optimizer_type = optimizer
        self.time_budget = time_budget
        self.each_run_budget = each_run_budget
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.memory_limit = memory_limit
        self.include_models = include_models
        self.exclude_models = exclude_models
        self.seed = seed
        self.tmp_dir = tmp_dir
        self.output_dir = output_dir
        self.pre_pipeline = None
        self._ml_engine = None

    def build_engine(self):
        engine = self.get_automl()(
            time_budget=self.time_budget,
            each_run_budget=self.each_run_budget,
            ensemble_method=self.ensemble_method,
            ensemble_size=self.ensemble_size,
            memory_limit=self.memory_limit,
            include_models=self.include_models,
            exclude_models=self.exclude_models,
            optimizer_type=self.optimizer_type,
            seed=self.seed
        )
        return engine

    def fit(self, data, **kwargs):
        assert data is not None and isinstance(data, (DataManager, pd.DataFrame))
        self._ml_engine = self.build_engine()
        self._ml_engine.fit(data, **kwargs)
        return self

    def predict(self, X, batch_size=None, n_jobs=1):
        return self._ml_engine.predict(X, batch_size=batch_size, n_jobs=n_jobs)

    def score(self, X, y):
        return self._ml_engine.score(X, y)

    def predict_proba(self, X, batch_size=None, n_jobs=1):
        return self._ml_engine.predict_proba(X, batch_size=None, n_jobs=1)

    def get_automl(self):
        raise NotImplementedError()

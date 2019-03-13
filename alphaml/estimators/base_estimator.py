class BaseEstimator(object):
    """Base class for all estimators in alpha-ml. """

    def __init__(
            self,
                 time_budget=3600,
                 each_run_budget=360,
                 ensemble_size=10,
                 memory_limit=1024,
                 include_models=None,
                 exclude_models=None,
                 tmp_dir=None,
                 output_dir=None):
        self.time_budget = time_budget
        self.each_run_budget = each_run_budget
        self.ensemble_size = ensemble_size
        self.memory_limit = memory_limit
        self.include_models = include_models
        self.exclude_models = exclude_models
        self.tmp_dir = tmp_dir
        self.output_dir = output_dir

        self._ml_engine = None

    def build_engine(self):
        engine = self.get_automl()(
            time_budget=self.time_budget,
            each_run_budget=self.each_run_budget,
            ensemble_size=self.ensemble_size,
            include_models=self.include_models,
            exclude_models=self.exclude_models
        )
        return engine

    def fit(self, **kwargs):
        self._ml_engine = self.build_engine()
        self._ml_engine.fit(**kwargs)
        return self

    def predict(self, X):
        return self._ml_engine.predict(X)

    def score(self, X, y):
        return self._ml_engine.score(X, y)

    def predict_proba(self, X):
        return self._ml_engine.predict_proba(X)

    def get_automl(self):
        raise NotImplementedError()

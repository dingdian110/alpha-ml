import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import mean_squared_error
from alphaml.estimators.base_estimator import BaseEstimator
from alphaml.engine.automl import AutoMLRegressor
from alphaml.engine.components.data_manager import DataManager
from alphaml.engine.components.pipeline.data_preprocessing_pipeline import DP_Pipeline
from alphaml.utils.metrics_util import get_metric


class Regressor(BaseEstimator):
    """This class implements the regression task. """

    def fit(self, data, **kwargs):
        """Fit the regressor to given training data.

        Parameters
        ----------

        data : instance of DataManager.

        metric : callable, optional (default='autosklearn.metrics.mean_squared_error').

        dataset_name : str, optional (default=None)
            Create nicer output. If None, a string will be determined by the
            md5 hash of the dataset.

        Returns
        -------
        self

        """
        # feat_type = None if 'feat_type' not in kwargs else kwargs['feat_type']
        # dataset_name = None if 'dataset_name' not in kwargs else kwargs['dataset_name']
        # # The number of evaluations.
        # runcount = None if 'runcount' not in kwargs else kwargs['runcount']

        # TODO:Automated feature engineering
        if isinstance(data, pd.DataFrame):
            self.pre_pipeline = DP_Pipeline(None)
            data = self.pre_pipeline.execute(data, phase='train', stratify=False)
        # Check the task type: {continuous}
        task_type = type_of_target(data.train_y)
        if task_type != 'continuous':
            raise ValueError("UNSUPPORTED TASK TYPE: %s!" % task_type)
        self.task_type = task_type
        kwargs['task_type'] = task_type

        metric = mean_squared_error if 'metric' not in kwargs else kwargs['metric']
        metric = get_metric(metric)
        kwargs['metric'] = metric

        super().fit(data, **kwargs)

        return self

    def predict(self, X, batch_size=None, n_jobs=1):
        """Predict classes for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features] or DataFrame

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_labels]
            The predicted classes.

        """
        if isinstance(X, pd.DataFrame):
            if not isinstance(self.pre_pipeline, DP_Pipeline):
                raise ValueError("The preprocessing pipeline is empty. Use DataFrame as the input of function fit.")
            dm = self.pre_pipeline.execute(X, phase='test')
            X = dm.test_X
        return super().predict(X, batch_size=batch_size, n_jobs=n_jobs)

    def get_automl(self):
        return AutoMLRegressor

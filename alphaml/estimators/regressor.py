import numpy as np
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import mean_squared_error
from alphaml.estimators.base_estimator import BaseEstimator
from alphaml.engine.automl import AutoMLRegressor
from alphaml.engine.components.data_manager import DataManager


class Regressor(BaseEstimator):
    """This class implements the classification task. """

    def fit(self, data, **kwargs):
        """Fit the classifier to given training data.

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

        metric = mean_squared_error if 'metric' not in kwargs else kwargs['metric']
        # feat_type = None if 'feat_type' not in kwargs else kwargs['feat_type']
        # dataset_name = None if 'dataset_name' not in kwargs else kwargs['dataset_name']
        # # The number of evaluations.
        # runcount = None if 'runcount' not in kwargs else kwargs['runcount']

        # Check the task type: {continuous}
        task_type = type_of_target(data.train_y)
        if task_type != 'continuous':
            raise ValueError("UNSUPPORTED TASK TYPE: %s!" % task_type)
        self.task_type = task_type
        kwargs['task_type'] = task_type
        assert data is not None and isinstance(data, DataManager)

        if isinstance(metric, str):
            from sklearn.metrics import mean_absolute_error, explained_variance_score, r2_score, mean_squared_log_error
            if metric == 'mse':
                metric = mean_squared_error
            elif metric == 'msle':
                metric = mean_squared_log_error
            elif metric == 'mae':
                metric = mean_absolute_error
            elif metric == 'evs':
                metric = explained_variance_score
            elif metric == 'r2':
                metric = r2_score
            else:
                raise ValueError('UNSUPPORTED metric: %s' % metric)
        if not hasattr(metric, '__call__'):
            raise ValueError('Input metric is not callable!')
        kwargs['metric'] = metric

        super().fit(data, **kwargs)

        return self

    def predict(self, X, batch_size=None, n_jobs=1):
        """Predict classes for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_labels]
            The predicted classes.

        """
        return super().predict(X, batch_size=batch_size, n_jobs=n_jobs)

    def get_automl(self):
        return AutoMLRegressor

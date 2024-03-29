import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import accuracy_score
from alphaml.estimators.base_estimator import BaseEstimator
from alphaml.engine.automl import AutoMLClassifier
from alphaml.engine.components.pipeline.data_preprocessing_pipeline import DP_Pipeline
from alphaml.utils.metrics_util import get_metric


class Classifier(BaseEstimator):
    """This class implements the classification task. """

    def fit(self, data, **kwargs):
        """Fit the classifier to given training data.

        Parameters
        ----------

        data : instance of DataManager or DataFrame

        metric : callable, optional (default='autosklearn.metrics.accuracy_score').

        feat_type : list, optional (default=None)
            List of str of `len(X.shape[1])` describing the attribute type.
            Possible types are `Categorical` and `Numerical`. `Categorical`
            attributes will be automatically One-Hot encoded. The values
            used for a categorical attribute must be integers, obtained for
            example by `sklearn.preprocessing.LabelEncoder
            <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html>`_.

        dataset_name : str, optional (default=None)
            Create nicer output. If None, a string will be determined by the
            md5 hash of the dataset.

        Returns
        -------
        self

        """

        metric = accuracy_score if 'metric' not in kwargs else kwargs['metric']
        # feat_type = None if 'feat_type' not in kwargs else kwargs['feat_type']
        # dataset_name = None if 'dataset_name' not in kwargs else kwargs['dataset_name']
        # # The number of evaluations.
        # runcount = None if 'runcount' not in kwargs else kwargs['runcount']

        # TODO:Automated feature engineering
        if isinstance(data, pd.DataFrame):
            self.pre_pipeline = DP_Pipeline(None)
            data = self.pre_pipeline.execute(data, phase='train')

        # Check the task type: {binary, multiclass}
        task_type = type_of_target(data.train_y)
        if task_type in ['multiclass-multioutput',
                         'continuous',
                         'continuous-multioutput',
                         'unknown']:
            raise ValueError("UNSUPPORTED TASK TYPE: %s!" % task_type)
        self.task_type = task_type
        kwargs['task_type'] = task_type

        # Options for multiclass averaging.
        average = 'weighted'

        metric = get_metric(metric)
        kwargs['metric'] = metric

        super().fit(data, **kwargs)

        return self

    def predict(self, X, batch_size=None, n_jobs=1):
        """Predict classes for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features] or Dataframe

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

    def predict_proba(self, X, batch_size=None, n_jobs=1):

        """Predict probabilities of classes for all samples X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features] or Dataframe

        batch_size : int (optional)
            Number of data points to predict for (predicts all points at once
            if ``None``.
        n_jobs : int

        Returns
        -------
        y : array of shape = [n_samples, n_classes]
            The predicted class probabilities.

        """
        if isinstance(X, pd.DataFrame):
            if not isinstance(self.pre_pipeline, DP_Pipeline):
                raise ValueError("The preprocessing pipeline is empty. Use DataFrame as the input of function fit.")
            dm = self.pre_pipeline.execute(X, phase='test')
            X = dm.test_X
        pred_proba = super().predict_proba(X, batch_size=batch_size, n_jobs=n_jobs)

        if self.task_type not in ['multilabel-indicator']:
            assert (
                np.allclose(
                    np.sum(pred_proba, axis=1),
                    np.ones_like(pred_proba[:, 0]))
            ), "prediction probability does not sum up to 1!"

        # Check that all probability values lie between 0 and 1.
        assert (
                (pred_proba >= 0).all() and (pred_proba <= 1).all()
        ), "found prediction probability value outside of [0, 1]!"

        return pred_proba

    def get_automl(self):
        return AutoMLClassifier

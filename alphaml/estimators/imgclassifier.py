import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import type_of_target
from alphaml.estimators.base_estimator import BaseEstimator
from alphaml.engine.automl import AutoIMGClassifier
from alphaml.engine.components.data_manager import DataManager


class ImageClassifier(BaseEstimator):
    """This class implements the classification task. """

    def fit(self,
            data,
            metric=accuracy_score,
            feat_type=None,
            dataset_name=None):
        """Fit the classifier to given training set (X, y).

        Fit both optimizes the machine learning models and builds an ensemble
        out of them. To disable ensembling, set ``ensemble_size==0``.

        Parameters
        ----------

        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target classes.

        X_test : array-like or sparse matrix of shape = [n_samples, n_features]
            Test data input samples. Will be used to save test predictions for
            all models. This allows to evaluate the performance of Auto-sklearn
            over time.

        y_test : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Test data target classes. Will be used to calculate the test error
            of all models. This allows to evaluate the performance of
            Auto-sklearn over time.

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
        # Check the task type: {binary, multiclass}
        task_type = type_of_target(data.train_y)
        if task_type in ['multiclass-multioutput',
                         'continuous',
                         'continuous-multioutput',
                         'unknown',
                         ]:
            raise ValueError("Problematic Task Type: %s!" % task_type)
        assert task_type in ['binary', 'multiclass', 'multilabel-indicator']
        assert data is not None and isinstance(data, DataManager)

        # Image classification task.
        task_type = 'img_' + task_type
        self.task_type = task_type

        super().fit(
            data,
            metric=metric,
            feat_type=feat_type,
            dataset_name=dataset_name,
            task_type=self.task_type
        )

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

    def predict_proba(self, X, batch_size=None, n_jobs=1):

        """Predict probabilities of classes for all samples X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        batch_size : int (optional)
            Number of data points to predict for (predicts all points at once
            if ``None``.
        n_jobs : int

        Returns
        -------
        y : array of shape = [n_samples, n_classes]
            The predicted class probabilities.

        """
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
        return AutoIMGClassifier

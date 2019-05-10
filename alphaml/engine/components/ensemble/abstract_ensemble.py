from abc import ABCMeta, abstractmethod


class AbstractEnsemble(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, val_x, val_y):
        """Fit an ensemble given a validation set and its label.

        Ensemble building maximizes performance (in contrast to
        hyperparameter optimization)!

        Parameters
        ----------
        val_x : array of shape = [n_data_points, n_features]

        val_y : label of each data point
        array of shape [n_data_points]

        ReturnsBlending.py
        -------
        self

        """
        pass

    @abstractmethod
    def predict(self, test_x):
        """Create ensemble predictions from the base model predictions.

        Parameters
        ----------
        test_x : array of shape = [n_data_points, n_features]
            Same as in the fit method.

        Returns
        -------
        array : [n_data_points]
        """
        self

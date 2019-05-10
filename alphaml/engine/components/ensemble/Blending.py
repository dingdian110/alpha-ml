import os
import sys
import warnings
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.utils.validation import check_X_y, check_array
from alphaml.engine.components.ensemble.abstract_ensemble import AbstractEnsemble



class Blending(AbstractEnsemble):
    """
    sample_weight:Individual weights for each sample
    regression : boolean, default True
        If True - perform stacking for regression task,
        if False - perform stacking for classification task
    val_zize : float, default 0.4
        split the train data into train data and validation data
    needs_proba: boolean, default False
        Whether to predict probabilities (instead of class labels)
    save_dir: str, default None
        a valid directory (must exist) where log  will be saved
    metric:callable, default None
        Evaluation metric (score function) which is used to calculate results of cross-validation.
    MEAN/FULL interpretation:
            MEAN - mean (average) of scores for each fold.
            FULL - metric calculated using combined oof predictions
                for full train set and target.
    n_folds : int, default 4
        Number of folds in cross-validation
    stratified : boolean, default False, meaningful only for classification task
        If True - use stratified folds in cross-validation
        Ignored if regression=True
    shuffle : boolean, default False
        Whether to perform a shuffle before cross-validation split
    random_state : int, default 0
        Random seed
    verbose : int, default 0
        Level of verbosity.
        0 - show no messages
        1 - for each model show mean score
        2 - for each model show score for each fold and mean score
    """

    def __init__(self, regression=False, needs_proba=False, metric=None, meta_learner=None, models=None,verbose = False):
        self.needs_proba = bool(needs_proba)
        if regression and needs_proba:
            warn_str = 'Task is regression <regression=True> hence function ignored classification-specific parameters which were set as <True>:'
            if needs_proba:
                self.needs_proba = False
                warn_str += ' <needs_proba>'
            warnings.warn(warn_str, UserWarning)
        if needs_proba and metric == 'accuracy':
            self.metric = log_loss
            warn_str = 'Task needs probability, so the metric is set to log_loss '
            warnings.warn(warn_str, UserWarning)
        self.metric = metric
        if metric is None and regression:
            self.metric = mean_absolute_error
        elif metric is None and not regression:
            if needs_proba:
                self.metric = log_loss
            else:
                self.metric = accuracy_score
        self.regression = bool(regression)
        self.models = models
        self.meta_learner = meta_learner
        self.n_classes = None
        self.action = None
        self.fitted_meta_learner = None
        self.verbose = verbose

    def fit(self, val_x, val_y):
        if self.verbose:
            if self.regression:
                task_str = 'task:       [regression]'
            else:
                task_str = 'task:       [classification]'
                n_classes_str = 'n_classes:  [%d]' % len(np.unique(val_y))
            metric_str = 'metric:     [%s]' % self.metric.__name__
            val_size_str = 'val_size:     [%s]' % val_y.size
            n_models_str = 'n_base_models:   [%d]' % len(self.models)
            print('-' * 40 + '\n')
            print(task_str)
            if not self.regression:
                print(n_classes_str)
            print(metric_str)
            print(val_size_str)
            print(n_models_str + '\n')
            print('-' * 40 + '\n')

        if not self.regression and self.needs_proba:
            self.n_classes = len(np.unique(val_y))
            self.action = 'predict_proba'
        else:
            self.n_classes = 1
            self.action = 'predict'
        val_predictions = []
        # for model in self.models:
        #     predictions = model.predict(val_x)
        #     val_predictions.append(predictions)

        ens_val = np.zeros((val_x.shape[0], len(self.models) * self.n_classes))
        for model_counter, model in enumerate(self.models):
            if 'predict_proba' == self.action:
                col_slice_model = slice(model_counter * self.n_classes, model_counter * self.n_classes + self.n_classes)
                ens_val[:, col_slice_model] = model.predict_proba(val_x)
            else:
                col_slice_model = model_counter
                ens_val[:, col_slice_model] = model.predict(val_x)
        self.fitted_meta_learner = self.meta_learner.fit(ens_val, val_y)

    def predict(self, test_x):
        ens_test = np.zeros((test_x.shape[0], len(self.models) * self.n_classes))
        for model_counter, model in enumerate(self.models):
            if 'predict_proba' == self.action:
                col_slice_model = slice(model_counter * self.n_classes, model_counter * self.n_classes + self.n_classes)
                ens_test[:, col_slice_model] = model.predict_proba(test_x)
            else:
                col_slice_model = model_counter
                ens_test[:, col_slice_model] = model.predict(test_x)
        pre_test = self.fitted_meta_learner.predict(ens_test)
        return pre_test


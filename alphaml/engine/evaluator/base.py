import time
import logging
import multiprocessing
import pickle as pkl
import os
from sklearn.metrics import roc_auc_score
from alphaml.engine.components.models.classification import _classifiers
from alphaml.engine.components.models.regression import _regressors
from alphaml.utils.save_ease import save_ease


def update_config(config):
    config_dict = {}
    for param in config:
        if param == 'estimator':
            continue
        if param.find(":") != -1:
            value = config[param]
            new_name = param.split(':')[-1]
            config_dict[new_name] = value
        else:
            config_dict[param] = config[param]
    return config_dict


class BaseClassificationEvaluator(object):
    """
    This Base Evaluator class must have: 1) data_manager and metric_func, 2) __call__ and fit_predict.
    """

    def __init__(self):
        self.data_manager = None
        self.metric_func = None
        self.logger = logging.getLogger(__name__)

    @save_ease(save_dir='./data/save_models')
    def __call__(self, config, **kwargs):
        # Build the corresponding estimator.
        classifier_type, estimator = self.set_config(config)
        save_path = kwargs['save_path']
        # TODO: how to parallize.
        if hasattr(estimator, 'n_jobs'):
            setattr(estimator, 'n_jobs', multiprocessing.cpu_count() - 1)
        start_time = time.time()
        self.logger.info('<START TO FIT> %s' % classifier_type)
        self.logger.info('<CONFIG> %s' % config)
        # Fit the estimator on the training data.
        estimator.fit(self.data_manager.train_X, self.data_manager.train_y)

        with open(save_path, 'wb') as f:
            self.logger.info('<MODEL SAVED IN %s>' % save_path)
            pkl.dump(estimator, f)

        # Validate it on val data.
        if self.metric_func == roc_auc_score:
            y_pred = estimator.predict_proba(self.data_manager.val_X)[:, 1]
            metric = self.metric_func(self.data_manager.val_y, y_pred)
        else:
            y_pred = estimator.predict(self.data_manager.val_X)
            metric = self.metric_func(self.data_manager.val_y, y_pred)

        self.logger.info('<EVALUATE %s TAKES %.2f SECONDS>' % (classifier_type, time.time() - start_time))
        # Turn it to a minimization problem.
        return 1 - metric

    def set_config(self, config):
        if not hasattr(self, 'estimator'):
            # Build the corresponding estimator.
            params_num = len(config.get_dictionary().keys()) - 1
            classifier_type = config['estimator']
            estimator = _classifiers[classifier_type](*[None] * params_num)
        else:
            estimator = self.estimator
        config = update_config(config)
        estimator.set_hyperparameters(config)
        return classifier_type, estimator

    @save_ease(save_dir='data/save_models')
    def fit_predict(self, config, test_X=None, **kwargs):
        # Build the corresponding estimator.
        save_path = kwargs['save_path']
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                estimator = pkl.load(f)
                print("Estimator loaded from", save_path)
        else:
            _, estimator = self.set_config(config)
            # Fit the estimator on the training data.
            estimator.fit(self.data_manager.train_X, self.data_manager.train_y)

        # Inference.
        if test_X is None:
            test_X = self.data_manager.test_X

        if self.metric_func == roc_auc_score:
            y_pred = estimator.predict_proba(test_X)[:, 1]
        else:
            y_pred = estimator.predict(test_X)
        return y_pred


class BaseRegressionEvaluator(object):
    """
    This Base Evaluator class must have: 1) data_manager and metric_func, 2) __call__ and fit_predict.
    """

    def __init__(self):
        self.data_manager = None
        self.metric_func = None
        self.logger = logging.getLogger(__name__)

    @save_ease(save_dir='./data/save_models')
    def __call__(self, config, **kwargs):
        # Build the corresponding estimator.
        regressor_type, estimator = self.set_config(config)
        save_path = kwargs['save_path']
        # TODO: how to parallize.
        if hasattr(estimator, 'n_jobs'):
            setattr(estimator, 'n_jobs', multiprocessing.cpu_count() - 1)
        start_time = time.time()
        self.logger.info('<START TO FIT> %s' % regressor_type)
        self.logger.info('<CONFIG> %s' % config)
        # Fit the estimator on the training data.
        estimator.fit(self.data_manager.train_X, self.data_manager.train_y)

        with open(save_path, 'wb') as f:
            pkl.dump(estimator, f)
            self.logger.info('<MODEL SAVED IN %s>' % save_path)

        # Validate it on val data.
        y_pred = estimator.predict(self.data_manager.val_X)
        metric = self.metric_func(self.data_manager.val_y, y_pred)

        self.logger.info('<EVALUATE %s TAKES %.2f SECONDS>' % (regressor_type, time.time() - start_time))
        # Turn it to a minimization problem.
        return metric

    def set_config(self, config):
        if not hasattr(self, 'estimator'):
            # Build the corresponding estimator.
            params_num = len(config.get_dictionary().keys()) - 1
            regressor_type = config['estimator']
            estimator = _regressors[regressor_type](*[None] * params_num)
        else:
            estimator = self.estimator
        config = update_config(config)
        estimator.set_hyperparameters(config)
        return regressor_type, estimator

    @save_ease(save_dir='data/save_models')
    def fit_predict(self, config, test_X=None, **kwargs):
        # Build the corresponding estimator.
        save_path = kwargs['save_path']
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                estimator = pkl.load(f)
                print("Estimator loaded from", save_path)
        else:
            _, estimator = self.set_config(config)
            # Fit the estimator on the training data.
            estimator.fit(self.data_manager.train_X, self.data_manager.train_y)

        # Inference.
        if test_X is None:
            test_X = self.data_manager.test_X
        y_pred = estimator.predict(test_X)
        return y_pred

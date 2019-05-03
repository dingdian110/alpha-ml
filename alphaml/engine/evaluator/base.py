import time
import logging
import multiprocessing
from alphaml.engine.components.models.classification import _classifiers


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


class BaseEvaluator(object):
    """
    This Base Evaluator class must have: 1) data_manager and metric_func, 2) __call__ and fit_predict.
    """

    def __init__(self):
        self.data_manager = None
        self.metric_func = None
        self.logger = logging.getLogger(__name__)

    def __call__(self, config):
        params_num = len(config.get_dictionary().keys()) - 1
        classifier_type = config['estimator']
        estimator = _classifiers[classifier_type](*[None]*params_num)
        config = update_config(config)
        estimator.set_hyperparameters(config)

        # TODO: how to parallize.
        if hasattr(estimator, 'n_jobs'):
            setattr(estimator, 'n_jobs', multiprocessing.cpu_count() - 1)
        start_time = time.time()
        self.logger.info('<START TO FIT> %s' % classifier_type)
        # Fit the estimator on the training data.
        estimator.fit(self.data_manager.train_X, self.data_manager.train_y)

        # Validate it on val data.
        y_pred = estimator.predict(self.data_manager.val_X)
        metric = self.metric_func(self.data_manager.val_y, y_pred)

        self.logger.info('<EVALUATE %s TAKES %.2f SECONDS>' % (classifier_type, time.time() - start_time))
        # Turn it to a minimization problem.
        return 1 - metric

    def fit_predict(self, config, test_X=None):
        if not hasattr(self, 'estimator'):
            # Build the corresponding estimator.
            params_num = len(config.get_dictionary().keys()) - 1
            classifier_type = config['estimator']
            estimator = _classifiers[classifier_type](*[None] * params_num)
        else:
            estimator = self.estimator
        config = update_config(config)
        estimator.set_hyperparameters(config)

        # Fit the estimator on the training data.
        estimator.fit(self.data_manager.train_X, self.data_manager.train_y)

        # Inference.
        if test_X is None:
            test_X = self.data_manager.test_X
        y_pred = estimator.predict(test_X)
        return y_pred

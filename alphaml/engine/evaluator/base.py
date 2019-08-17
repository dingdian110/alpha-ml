import time
import logging
import multiprocessing
import pickle as pkl
from alphaml.engine.components.models.classification import _classifiers
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


class BaseEvaluator(object):
    """
    This Base Evaluator class must have: 1) data_manager and metric_func, 2) __call__ and fit_predict.
    """

    def __init__(self):
        self.data_manager = None
        self.metric_func = None
        self.logger = logging.getLogger(__name__)

    def get_config(self, config):
        params_num = len(config.get_dictionary().keys()) - 1
        classifier_type = config['estimator']
        estimator = _classifiers[classifier_type](*[None] * params_num)
        config = update_config(config)
        estimator.set_hyperparameters(config)
        return classifier_type, estimator

    @save_ease(save_dir='data/save_models')
    def __call__(self, config, **kwargs):
        classifier_type, estimator = self.get_config(config)
        save_path = kwargs['save_path']

        # TODO: how to parallize.
        if hasattr(estimator, 'n_jobs'):
            setattr(estimator, 'n_jobs', multiprocessing.cpu_count() - 1)
        start_time = time.time()
        self.logger.info('<START TO FIT> %s' % classifier_type)
        self.logger.info('<CONFIG> %s' % config.get_dictionary())
        # Fit the estimator on the training data.
        estimator.fit(self.data_manager.train_X, self.data_manager.train_y)

        with open(save_path, 'wb') as f:
            pkl.dump(estimator, f)

        # Validate it on val data.
        y_pred = estimator.predict(self.data_manager.val_X)
        metric = self.metric_func(self.data_manager.val_y, y_pred)

        self.logger.info('<EVALUATE %s TAKES %.2f SECONDS>' % (classifier_type, time.time() - start_time))
        # Turn it to a minimization problem.
        return 1 - metric

    def fit_predict(self, config, test_X=None):
        classifier_type, estimator = self.get_config(config)
        # Fit the estimator on the training data.
        estimator.fit(self.data_manager.train_X, self.data_manager.train_y)

        # Inference.
        if test_X is None:
            test_X = self.data_manager.test_X
        y_pred = estimator.predict(test_X)
        return y_pred

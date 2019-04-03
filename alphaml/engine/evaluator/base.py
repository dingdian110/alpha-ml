from alphaml.engine.components.models.classification import _classifiers


def update_config(config):
    config_dict = {}
    for param in config:
        if param.find(":") != -1:
            value = config[param]
            new_name = param.split(':')[-1]
            config_dict[new_name] = value
    return config_dict


class BaseEvaluator(object):
    def __init__(self, data, metric):
        self.data_manager = data
        self.metric_func = metric

    def __call__(self, config):
        params_num = len(config.get_dictionary().keys()) - 1
        classifier_type = config['estimator']
        estimator = _classifiers[classifier_type](*[None]*params_num)
        config = update_config(config)
        estimator.set_hyperparameters(config)

        # Fit the estimator on the training data.
        estimator.fit(self.data_manager.train_X, self.data_manager.train_y)

        # Validate it on val data.
        y_pred = estimator.predict(self.data_manager.val_X)
        metric = self.metric_func(self.data_manager.val_y, y_pred)

        # Turn it to a minimization problem.
        return 1 - metric


class HPOEvaluator(object):
    def __init__(self, data, metric, estimator):
        self.data_manager = data
        self.metric_func = metric
        self.estimator = estimator

    def __call__(self, config):
        self.estimator.set_hyperparameters(config.get_dictionary())

        # Fit the estimator on the training data.
        self.estimator.fit(self.data_manager.train_X, self.data_manager.train_y)

        # Validate it on val data.
        y_pred = self.estimator.predict(self.data_manager.val_X)
        metric = self.metric_func(self.data_manager.val_y, y_pred)

        # Turn it to a minimization problem.
        return 1 - metric

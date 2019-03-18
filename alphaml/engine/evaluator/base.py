import sklearn
from alphaml.engine.components.models.classification import _classifiers


def update_config(config):
    config_dict = {}
    for param in config:
        if param.find(":") != -1:
            value = config[param]
            new_name = param.split(':')[-1]
            config_dict[new_name] = value
    return config_dict


class SimpleEvaluator():
    def __init__(self):
        X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_test
        self.y_val = y_test
        self.metric_func = sklearn.metrics.accuracy_score

    def __call__(self, config):
        params_num = len(config.get_dictionary().keys()) - 1
        classifier_type = config['classifier']
        estimator = _classifiers[classifier_type](*[None]*params_num)
        config = update_config(config)
        estimator.set_hyperparameters(config)
        estimator.fit(self.X_train, self.y_train)
        y_pred = estimator.predict(self.X_val)
        acc = self.metric_func(y_pred, self.y_val)
        return acc

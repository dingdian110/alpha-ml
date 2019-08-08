from alphaml.utils.common import get_max_index
from alphaml.engine.evaluator.base import BaseClassificationEvaluator, BaseRegressionEvaluator
from alphaml.utils.save_ease import save_ease

import os
import pickle as pkl

CLASSIFICATION = 1
REGRESSION = 2


class BaseEnsembleModel(object):
    def __init__(self, model_info, ensemble_size, task_type, model_type='ml'):
        self.model_info = model_info
        self.model_type = model_type
        self.ensemble_models = list()
        if task_type in ['binary', 'multiclass', 'img_binary', 'img_multiclass', 'img_multilabel-indicator']:
            self.task_type = CLASSIFICATION
        elif task_type in ['continuous']:
            self.task_type = REGRESSION
        else:
            raise ValueError('Undefined Task Type: %s' % task_type)

        if len(model_info[0]) < ensemble_size:
            self.ensemble_size = len(model_info[0])
        else:
            self.ensemble_size = ensemble_size

        # Determine the best basic models (with the best performance) from models_infos.
        index_list = get_max_index(self.model_info[1], self.ensemble_size)
        self.config_list = [self.model_info[0][i] for i in index_list]
        for i in index_list:
            print(self.model_info[0][i], self.model_info[1][i])

    def fit(self, dm):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    @save_ease(save_dir='./data/save_models')
    def get_estimator(self, config, x, y, if_load=False, **kwargs):
        save_path = kwargs['save_path']
        if if_load and os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                estimator = pkl.load(f)
                print("Estimator loaded from", save_path)
        else:
            if self.task_type == CLASSIFICATION:
                evaluator = BaseClassificationEvaluator()
            elif self.task_type == REGRESSION:
                evaluator = BaseRegressionEvaluator()
            _, estimator = evaluator.set_config(config)
            estimator.fit(x, y)
            print("Estimator retrained.")
        return estimator

    def get_predictions(self, estimator, X):
        if self.task_type == CLASSIFICATION:
            return estimator.predict_proba(X)
        elif self.task_type == REGRESSION:
            pred = estimator.predict(X)
            shape = pred.shape
            if len(shape) == 1:
                pred = pred.reshape((shape[0], 1))
            return pred

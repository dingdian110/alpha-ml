from alphaml.utils.constants import *
from alphaml.engine.evaluator.base import BaseClassificationEvaluator, BaseRegressionEvaluator
from alphaml.engine.evaluator.hyperopt_evaluator import HyperoptClassificationEvaluator
from alphaml.utils.save_ease import save_ease

import os
import pickle as pkl
import functools
import math
import logging


class BaseEnsembleModel(object):
    def __init__(self, model_info, ensemble_size, task_type, metric, evaluator, model_type='ml', threshold=0.2):
        self.model_info = model_info
        self.model_type = model_type
        self.metric = metric
        self.evaluator = evaluator
        self.ensemble_models = list()
        self.threshold = threshold
        self.logger = logging.getLogger()
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
        # index_list = get_max_index(self.model_info[1], self.ensemble_size)

        # Determine the best basic models (the best for each algorithm) from models_infos.
        index_list = []
        model_len = len(self.model_info[1])

        def cmp(x, y):
            if self.model_info[1][x] > self.model_info[1][y]:
                return -1
            elif self.model_info[1][x] == self.model_info[1][y]:
                return 0
            else:
                return 1

        best_performance = float('-INF')
        try:
            # SMAC
            estimator_set = set([self.model_info[0][i]['estimator'] for i in range(model_len)])
            top_k = math.ceil(ensemble_size / len(estimator_set))
            # Get the estimator with the best performance for each algorithm
            for estimator in estimator_set:
                id_list = []
                for i in range(model_len):
                    if self.model_info[0][i]['estimator'] == estimator:
                        if self.model_info[1][i] != FAILED:
                            if best_performance < self.model_info[1][i]:
                                best_performance = self.model_info[1][i]
                            id_list.append(i)
                sort_list = sorted(id_list, key=functools.cmp_to_key(cmp))
                index_list.extend(sort_list[:top_k])

        except:
            # Hyperopt
            estimator_set = set(self.model_info[0][i]['estimator'][0] for i in range(model_len))
            top_k = math.ceil(ensemble_size / len(estimator_set))
            for estimator in estimator_set:
                id_list = []
                for i in range(model_len):
                    if self.model_info[0][i]['estimator'][0] == estimator:
                        if self.model_info[1][i] != FAILED:
                            if best_performance < self.model_info[1][i]:
                                best_performance = self.model_info[1][i]
                            id_list.append(i)
                sort_list = sorted(id_list, key=functools.cmp_to_key(cmp))
                index_list.extend(sort_list[:top_k])

        self.config_list = []
        for i in index_list:
            if abs((best_performance - self.model_info[1][i]) / best_performance) < self.threshold:
                self.logger.info('------------------')
                self.config_list.append(self.model_info[0][i])
                self.logger.info(str(self.model_info[0][i]))
                self.logger.info("Valid performance: " + str(self.model_info[1][i]))
                self.get_estimator(self.model_info[0][i], None, None, if_show=True)
                self.logger.info('------------------')

    def fit(self, dm):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_each(self, X):
        raise NotImplementedError

    @save_ease(save_dir='./data/save_models')
    def get_estimator(self, config, x, y, if_load=False, if_show=False, **kwargs):
        save_path = kwargs['save_path']
        if if_show:
            self.logger.info("Estimator path: " + save_path)
            return None
        if if_load and os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                estimator = pkl.load(f)
                self.logger.info("Estimator loaded from " + save_path)
        else:
            save_path = kwargs['save_path']
            _, estimator = self.evaluator.set_config(config)
            estimator.fit(x, y)
            with open(save_path, 'wb') as f:
                pkl.dump(estimator, f)
                self.logger.info("Estimator retrained!")
        return estimator

    def get_proba_predictions(self, estimator, X):
        if self.task_type == CLASSIFICATION:
            return estimator.predict_proba(X)
        elif self.task_type == REGRESSION:
            pred = estimator.predict(X)
            shape = pred.shape
            if len(shape) == 1:
                pred = pred.reshape((shape[0], 1))
            return pred

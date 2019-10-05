from alphaml.utils.common import get_max_index
from alphaml.engine.evaluator.base import BaseClassificationEvaluator, BaseRegressionEvaluator
from alphaml.engine.evaluator.hyperopt_evaluator import HyperoptClassificationEvaluator
from alphaml.utils.save_ease import save_ease

import os
import pickle as pkl
import functools
import math

CLASSIFICATION = 1
REGRESSION = 2
HYPEROPT_CLASSIFICATION = 3

FAILED = -2147483646.0


class BaseEnsembleModel(object):
    def __init__(self, model_info, ensemble_size, task_type, metric, model_type='ml'):
        self.model_info = model_info
        self.model_type = model_type
        self.metric = metric
        self.ensemble_models = list()
        if task_type in ['binary', 'multiclass', 'img_binary', 'img_multiclass', 'img_multilabel-indicator']:
            self.task_type = CLASSIFICATION
        elif task_type in ['continuous']:
            self.task_type = REGRESSION
        elif 'hyperopt' in task_type:
            self.task_type = HYPEROPT_CLASSIFICATION
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
                            id_list.append(i)
                sort_list = sorted(id_list, key=functools.cmp_to_key(cmp))
                index_list.extend(sort_list[:top_k])
            self.config_list = [self.model_info[0][i] for i in index_list]
        except:
            # Hyperopt
            estimator_set = set(self.model_info[0][i]['estimator'][0] for i in range(model_len))
            top_k = math.ceil(model_len / len(estimator_set))
            for estimator in estimator_set:
                id_list = []
                for i in range(model_len):
                    if self.model_info[0][i]['estimator'][0] == estimator:
                        id_list.append(i)
                sort_list = sorted(id_list, key=functools.cmp_to_key(cmp))
                index_list.extend(sort_list[:top_k])

            self.config_list = [self.model_info[0][i] for i in index_list]

        # print(self.model_info)
        for i in index_list:
            print('------------------')
            print(self.model_info[0][i], self.model_info[1][i])
            # self.get_estimator(self.model_info[0][i], None, None, True)
            print('------------------')

    def fit(self, dm):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    @save_ease(save_dir='./data/save_models')
    def get_estimator(self, config, x, y, if_load=False, **kwargs):
        save_path = kwargs['save_path']
        estimator_name = config['estimator']
        if isinstance(estimator_name, tuple):
            estimator_name = estimator_name[0]
        if if_load and os.path.exists(save_path) and estimator_name != 'xgboost':
            with open(save_path, 'rb') as f:
                estimator = pkl.load(f)
                print("Estimator loaded from", save_path)
        else:
            if self.task_type == CLASSIFICATION:
                evaluator = BaseClassificationEvaluator()
            elif self.task_type == REGRESSION:
                evaluator = BaseRegressionEvaluator()
            elif self.task_type == HYPEROPT_CLASSIFICATION:
                evaluator = HyperoptClassificationEvaluator()
            _, estimator = evaluator.set_config(config)
            estimator.fit(x, y)
            print("Estimator retrained.")
        return estimator

    def get_proba_predictions(self, estimator, X):
        if self.task_type == CLASSIFICATION or HYPEROPT_CLASSIFICATION:
            from sklearn.metrics import roc_auc_score
            if self.metric == roc_auc_score:
                return estimator.predict_proba(X)[:, 1:2]
            else:
                return estimator.predict_proba(X)
        elif self.task_type == REGRESSION:
            pred = estimator.predict(X)
            shape = pred.shape
            if len(shape) == 1:
                pred = pred.reshape((shape[0], 1))
            return pred

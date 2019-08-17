from alphaml.utils.common import get_max_index
from alphaml.engine.evaluator.base import BaseClassificationEvaluator, BaseRegressionEvaluator
from alphaml.utils.save_ease import save_ease

import os
import pickle as pkl

CLASSIFICATION = 1
REGRESSION = 2


class BaseEnsembleModel(object):
    def __init__(self, model_info, ensemble_size, task_type, metric, model_type='ml'):
        self.model_info = model_info
        self.model_type = model_type
        self.metric = metric
        self.ensemble_models = list()
        if 'hyperopt' in task_type:
            task_type = task_type.split('_')[1]
        if task_type in ['binary', 'multiclass', 'img_binary', 'img_multiclass', 'img_multilabel-indicator']:
            self.task_type = CLASSIFICATION
        elif task_type in ['continuous']:
            self.task_type = REGRESSION
        elif 'hyperopt' in task_type:
            self.task_type = CLASSIFICATION
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
        try:
            # SMAC
            estimator_set = set([self.model_info[0][i]['estimator'] for i in range(model_len)])
            # Get the estimator with the best performance for each algorithm
            for estimator in estimator_set:
                best_perf = -float("Inf")
                best_id = -1
                for i in range(model_len):
                    if self.model_info[0][i]['estimator'] == estimator:
                        if self.model_info[1][i] > best_perf:
                            best_perf = self.model_info[1][i]
                            best_id = i
                index_list.append(best_id)
        except:
            # Hyperopt
            estimator_set = set(self.model_info[0][i]['estimator'][0] for i in range(model_len))
            for estimator in estimator_set:
                best_perf = -float("Inf")
                best_id = -1
                for i in range(model_len):
                    if self.model_info[0][i]['estimator'][0] == estimator:
                        if self.model_info[1][i] > best_perf:
                            best_perf = self.model_info[1][i]
                            best_id = i
                index_list.append(best_id)

        self.config_list = [self.model_info[0][i] for i in index_list]
        for i in index_list:
            print('------------------')
            print(self.model_info[0][i], self.model_info[1][i])
            self.get_estimator(self.model_info[0][i], None, None, True)
            print('------------------')

    def fit(self, dm):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    @save_ease(save_dir='./data/save_models')
    def get_estimator(self, config, x, y, if_load=False, **kwargs):
        save_path = kwargs['save_path']
        if config['estimator'] != 'xgboost' and if_load and os.path.exists(save_path):
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

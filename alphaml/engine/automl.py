import logging
from alphaml.engine.components.components_manager import ComponentsManager
from alphaml.engine.components.data_manager import DataManager
from alphaml.engine.evaluator.base import BaseEvaluator
from alphaml.engine.optimizer.smac_smbo import SMAC_SMBO
from alphaml.engine.optimizer.ts_smbo import TS_SMBO
from alphaml.utils.label_util import to_categorical, map_label, get_classnum
import numpy as np


class AutoML(object):
    def __init__(
            self,
            time_budget,
            each_run_budget,
            memory_limit,
            ensemble_size,
            include_models,
            exclude_models,
            optimizer_type,
            random_seed=42):
        self.time_budget = time_budget
        self.each_run_budget = each_run_budget
        self.ensemble_size = ensemble_size
        self.memory_limit = memory_limit
        self.include_models = include_models
        self.exclude_models = exclude_models
        self.component_manager = ComponentsManager()
        self.optimizer_type = optimizer_type
        self.seed = random_seed
        self.optimizer = None
        self.evaluator = None
        self.metric = None
        self.logger = logging.getLogger(__name__)

    def fit(self, data: DataManager, **kwargs):
        """
        1. Define the ML pipeline.
           1) the configuration space.

        2. Search the promising configurations for this pipeline.

        :param X: array-like or sparse matrix of shape = [n_samples, n_features], the training input samples.
        :param y: array-like, shape = [n_samples], the training target.
        :return: self
        """

        task_type = kwargs['task_type']
        self.metric = kwargs['metric']

        # Get the configuration space for the automl task.
        config_space = self.component_manager.get_hyperparameter_search_space(
            task_type, self.include_models, self.exclude_models)

        self.logger.debug('The optimizer type is: %s' % self.optimizer_type)
        if self.optimizer_type == 'smbo':
            # Create optimizer.
            self.optimizer = SMAC_SMBO(self.evaluator, config_space, data, self.seed, **kwargs)
            self.optimizer.run()
        elif self.optimizer_type == 'ts_smbo':
            # Create optimizer.
            self.optimizer = TS_SMBO(self.evaluator, config_space, data, self.seed, **kwargs)
            self.optimizer.run()
        else:
            raise ValueError('UNSUPPORTED optimizer: %s' % self.optimizer)
        return self

    def predict(self, X, **kwargs):
        # For traditional ML task:
        #   fit the optimized model on the whole training data and predict the input data's labels.
        pred = self.evaluator.fit_predict(self.optimizer.incumbent, X)
        return pred

    def score(self, X, y):
        pred_y = self.predict(X)
        score = self.metric(y, pred_y)
        return score


class AutoMLClassifier(AutoML):
    def __init__(self,
                 time_budget,
                 each_run_budget,
                 memory_limit,
                 ensemble_size,
                 include_models,
                 exclude_models,
                 optimizer_type,
                 random_seed=42):
        super().__init__(time_budget, each_run_budget, memory_limit, ensemble_size, include_models,
                         exclude_models, optimizer_type, random_seed)
        self.evaluator = BaseEvaluator()

    def fit(self, data, **kwargs):
        return super().fit(data, **kwargs)


class AutoIMGClassifier(AutoML):
    def __init__(self,
                 time_budget,
                 each_run_budget,
                 memory_limit,
                 ensemble_size,
                 include_models,
                 exclude_models,
                 optimizer_type,
                 random_seed=42):
        super().__init__(time_budget, each_run_budget, memory_limit, ensemble_size, include_models,
                         exclude_models, optimizer_type, random_seed)

        self.evaluator = None
        self.map_dict = None
        self.rev_map_dict = None

    def fit(self, data: DataManager, **kwargs):
        from alphaml.engine.evaluator.dl_evaluator import BaseImgEvaluator
        task_type = kwargs['task_type']
        inputshape = data.train_X.shape[1:]
        classnum = None
        if task_type == 'img_multiclass':
            data.train_y, self.map_dict, self.rev_map_dict = map_label(data.train_y)
            data.train_y = to_categorical(data.train_y)
            data.val_y, _, _ = map_label(data.val_y, self.map_dict)
            data.val_y = to_categorical(data.val_y)
            classnum = len(self.rev_map_dict)

        elif task_type == 'img_binary':
            data.train_y, self.map_dict, self.rev_map_dict = map_label(data.train_y, if_binary=True)
            data.val_y, _, _ = map_label(data.val_y, self.map_dict, if_binary=True)
            classnum = 1

        elif task_type == 'img_multilabel-indicator':
            classnum = get_classnum(data.train_y)

        self.evaluator = BaseImgEvaluator(inputshape, classnum)

        return super().fit(data, **kwargs)

    def predict(self, X, **kwargs):
        y_pred = super().predict(X, **kwargs)
        if self.rev_map_dict is not None:
            y_pred = np.argmax(y_pred, axis=-1)
            y_pred = [self.rev_map_dict[i] for i in y_pred]
            y_pred = np.array(y_pred)
        return y_pred

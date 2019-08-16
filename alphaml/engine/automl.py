import logging
from alphaml.engine.components.components_manager import ComponentsManager
from alphaml.engine.components.data_manager import DataManager
from alphaml.engine.evaluator.base import BaseClassificationEvaluator, BaseRegressionEvaluator
from alphaml.engine.evaluator.hyperopt_evaluator import HyperoptClassificationEvaluator
from alphaml.engine.optimizer.smac_smbo import SMAC_SMBO
from alphaml.engine.optimizer.ts_smbo import TS_SMBO
from alphaml.engine.optimizer.nonstationary_mab_optimizer import TS_NON_SMBO
from alphaml.engine.optimizer.monotone_mab_optimizer import MONO_MAB_SMBO
from alphaml.engine.optimizer.cmab_optimizer import CMAB_TS
from alphaml.engine.optimizer.baseline_optimizer import BASELINE
from alphaml.engine.optimizer.sh_optimizer import SH_SMBO
from alphaml.engine.optimizer.rl_optimizer import RL_SMBO
from alphaml.engine.optimizer.mcmc_ts_optimizer import MCMC_TS_Optimizer
from alphaml.engine.optimizer.ucb_mab_optimizer import UCB_SMBO
from alphaml.engine.optimizer.hyperopt import Hyperopt
from alphaml.engine.components.ensemble.bagging import Bagging
from alphaml.engine.components.ensemble.blending import Blending
from alphaml.engine.components.ensemble.stacking import Stacking
from alphaml.engine.components.ensemble.ensemble_selection import EnsembleSelection
from alphaml.utils.label_util import to_categorical, map_label, get_classnum
import numpy as np
import pandas as pd


class AutoML(object):
    def __init__(
            self,
            time_budget,
            each_run_budget,
            memory_limit,
            ensemble_method,
            ensemble_size,
            include_models,
            exclude_models,
            optimizer_type,
            seed=42):
        self.time_budget = time_budget
        self.each_run_budget = each_run_budget
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.memory_limit = memory_limit
        self.include_models = include_models
        self.exclude_models = exclude_models
        self.component_manager = ComponentsManager()
        self.optimizer_type = optimizer_type

        self.seed = seed
        self.optimizer = None
        self.evaluator = None
        self.metric = None
        self.logger = logging.getLogger(__name__)
        self.ensemble_model = None

    def fit(self, data, **kwargs):
        """
        1. Define the ML pipeline.
           1) the configuration space.

        2. Search the promising configurations for this pipeline.

        :param data: A DataManager or a DataFrame. Automated feature engineering will be applied if data is a DataFrame
        :return: self
        """

        task_type = kwargs['task_type']
        self.metric = kwargs['metric']
        print(data.train_X.shape[1])
        # Get the configuration space for the automl task.
        config_space = self.component_manager.get_hyperparameter_search_space(
            task_type, self.include_models, self.exclude_models)

        self.logger.debug('The optimizer type is: %s' % self.optimizer_type)

        # Conduct algorithm selection and hyperparameter optimization.
        if self.optimizer_type == 'smbo':
            # Create optimizer.
            self.optimizer = SMAC_SMBO(self.evaluator, config_space, data, self.seed, **kwargs)
            self.optimizer.run()
        elif self.optimizer_type == 'ts_smbo':
            # Create optimizer.
            self.optimizer = TS_SMBO(self.evaluator, config_space, data, self.seed, **kwargs)
            self.optimizer.run()
        elif self.optimizer_type == 'non_smbo':
            # Create optimizer.
            self.optimizer = TS_NON_SMBO(self.evaluator, config_space, data, self.seed, **kwargs)
            self.optimizer.run()
        elif self.optimizer_type == 'mono_smbo':
            # Create optimizer.
            self.optimizer = MONO_MAB_SMBO(self.evaluator, config_space, data, self.seed, **kwargs)
            self.optimizer.run()
        elif self.optimizer_type == 'cmab_ts':
            # Create optimizer.
            self.optimizer = CMAB_TS(self.evaluator, config_space, data, self.seed, **kwargs)
            self.optimizer.run()
        elif self.optimizer_type == 'baseline':
            # Create optimizer.
            self.optimizer = BASELINE(self.evaluator, config_space, data, self.seed, **kwargs)
            self.optimizer.run()
        elif self.optimizer_type == 'sh':
            # Create optimizer.
            self.optimizer = SH_SMBO(self.evaluator, config_space, data, self.seed, **kwargs)
            self.optimizer.run()
        elif self.optimizer_type == 'rl_smbo':
            # Create optimizer.
            self.optimizer = RL_SMBO(self.evaluator, config_space, data, self.seed, **kwargs)
            self.optimizer.run()
        elif self.optimizer_type == 'mcmc_ts_smbo':
            # Create optimizer.
            self.optimizer = MCMC_TS_Optimizer(self.evaluator, config_space, data, self.seed, **kwargs)
            self.optimizer.run()
        elif self.optimizer_type == 'ucb_smbo':
            # Create optimizer.
            self.optimizer = UCB_SMBO(self.evaluator, config_space, data, self.seed, **kwargs)
            self.optimizer.run()
        elif self.optimizer_type == 'tpe':
            self.evaluator = HyperoptClassificationEvaluator()
            task_type = 'hyperopt_' + task_type
            config_space = self.component_manager.get_hyperparameter_search_space(
                task_type, self.include_models, self.exclude_models)
            self.optimizer = Hyperopt(self.evaluator, config_space, data, self.seed, **kwargs)
            self.optimizer.run()
        else:
            raise ValueError('UNSUPPORTED optimizer: %s' % self.optimizer)

        # Construct the ensemble model according to the ensemble method.
        model_infos = (self.optimizer.configs_list, self.optimizer.config_values)
        if self.ensemble_method == 'none':
            self.ensemble_model = None
        else:
            if self.ensemble_method == 'bagging':
                self.ensemble_model = Bagging(model_infos, self.ensemble_size, task_type, self.metric)
            elif self.ensemble_method == 'blending':
                self.ensemble_model = Blending(model_infos, self.ensemble_size, task_type, self.metric)
            elif self.ensemble_method == 'stacking':
                self.ensemble_model = Stacking(model_infos, self.ensemble_size, task_type, self.metric)
            elif self.ensemble_method == 'ensemble_selection':
                self.ensemble_model = EnsembleSelection(model_infos, self.ensemble_size, task_type, self.metric)
            else:
                raise ValueError('UNSUPPORTED ensemble method: %s' % self.ensemble_method)

        if self.ensemble_model is not None:
            # Train the ensemble model.
            self.ensemble_model.fit(data)

        return self

    def predict(self, X, **kwargs):
        print(X.shape[1])
        if self.ensemble_model is None:
            # For traditional ML task:
            #   fit the optimized model on the whole training data and predict the input data's labels.
            pred = self.evaluator.fit_predict(self.optimizer.incumbent, X)
            return pred
        else:
            # Predict the result.
            pred1 = self.ensemble_model.predict(X)
            return pred1

    def score(self, X, y):
        pred_y = self.predict(X)
        score = self.metric(y, pred_y)
        return score


class AutoMLClassifier(AutoML):
    def __init__(self,
                 time_budget,
                 each_run_budget,
                 memory_limit,
                 ensemble_method,
                 ensemble_size,
                 include_models,
                 exclude_models,
                 optimizer_type,
                 seed=None):
        super().__init__(time_budget, each_run_budget, memory_limit, ensemble_method, ensemble_size, include_models,
                         exclude_models, optimizer_type, seed)
        self.evaluator = BaseClassificationEvaluator()

    def fit(self, data, **kwargs):
        return super().fit(data, **kwargs)


class AutoMLRegressor(AutoML):
    def __init__(self,
                 time_budget,
                 each_run_budget,
                 memory_limit,
                 ensemble_method,
                 ensemble_size,
                 include_models,
                 exclude_models,
                 optimizer_type,
                 seed=None):
        super().__init__(time_budget, each_run_budget, memory_limit, ensemble_method, ensemble_size, include_models,
                         exclude_models, optimizer_type, seed)
        self.evaluator = BaseRegressionEvaluator()

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
                 seed=None):
        super().__init__(time_budget, each_run_budget, memory_limit, ensemble_size, include_models,
                         exclude_models, optimizer_type, seed)

        self.evaluator = None
        self.map_dict = None
        self.rev_map_dict = None

    def fit(self, data: DataManager, **kwargs):
        from alphaml.engine.evaluator.dl_evaluator import BaseImgEvaluator
        task_type = kwargs['task_type']
        if data.train_X is None and data.train_y is None:
            inputshape = data.target_shape
            classnum = None
        else:
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

    def score(self, X, y):
        raise NotImplementedError()

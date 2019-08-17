import time
import pickle
import numpy as np
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, space_eval
from alphaml.engine.optimizer.base_optimizer import BaseOptimizer


class Hyperopt(BaseOptimizer):
    def __init__(self, evaluator, config_space, data, seed, **kwargs):
        super().__init__(evaluator, config_space, data, kwargs['metric'], seed)
        self.task_name = kwargs['task_name'] if 'task_name' in kwargs else 'default'
        self.result_file = self.task_name + '_hyperopt.data'
        self.estimators = list(self.config_space.keys())
        self.config_space = {
            'estimator': hp.choice('estimator',
                                   [(estimator, self.config_space[estimator]) for estimator in self.estimators])}
        self.trials = Trials()
        self.runcount = int(1e10) if 'runcount' not in kwargs or kwargs is None else kwargs['runcount']

        def objective(x):
            return {
                'loss': self.evaluator(x),
                'status': STATUS_OK,
                'config': x
            }

        self.objective = objective
        self.configs_list = []
        self.config_values = []

    def run(self):
        self.logger.info('Start task: %s' % self.task_name)

        fmin(self.objective, self.config_space, tpe.suggest, self.runcount, trials=self.trials)
        best = -float('inf')
        for trial in self.trials.trials:
            config = trial['result']['config']
            self.configs_list.append(config)
            result = 1 - trial['result']['loss']
            if result > best:
                best = result
                self.incumbent = config
            self.config_values.append(1 - trial['result']['loss'])

        self.logger.info('TPE ==> the size of evaluations: %d' % len(self.configs_list))
        if len(self.configs_list) > 0:
            self.logger.info('TPE ==> The time points: %s' % self.timing_list)
            self.logger.info('TPE ==> The best performance found: %f' % max(self.config_values))
            self.logger.info('TPE ==> The best HP found: %s' % self.incumbent)

            # Save the experimental results.
            data = dict()
            data['configs'] = self.configs_list
            data['perfs'] = self.config_values
            dataset_id = self.result_file.split('_')[0]
            with open('data/%s/' % dataset_id + self.result_file, 'wb') as f:
                pickle.dump(data, f)

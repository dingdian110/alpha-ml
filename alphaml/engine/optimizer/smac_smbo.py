import time
import pickle
import numpy as np
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from alphaml.engine.optimizer.base_optimizer import BaseOptimizer


class SMAC_SMBO(BaseOptimizer):
    def __init__(self, evaluator, config_space, data, seed, **kwargs):
        super().__init__(evaluator, config_space, data, kwargs['metric'], seed)
        self.task_name = kwargs['task_name'] if 'task_name' in kwargs else 'default'
        self.result_file = self.task_name + '_smac.data'

        # Scenario object
        scenario_dict = {
            'abort_on_first_run_crash': True,
            "run_obj": "quality",
            "cs": self.config_space,
            "deterministic": "true"
        }
        if 'runcount' in kwargs and kwargs['runcount'] > 0:
            scenario_dict['runcount-limit'] = kwargs['runcount']
        self.scenario = Scenario(scenario_dict)
        self.smac = SMAC(scenario=self.scenario, rng=np.random.RandomState(self.seed), tae_runner=self.evaluator)
<<<<<<< HEAD
        self.configs_list = list()
        self.config_values = list()

    def run(self):
=======
        self.configs_list=list()
        self.config_values=list()

    def run(self):
        # configs_list = list()
        # config_values = list()
        time_list = list()
        start_time = time.time()
>>>>>>> 8f371cf61c8fb225ebfc75059dc361026e8d96e7
        self.logger.info('Start task: %s' % self.task_name)

        self.smac.optimize()
        runhistory = self.smac.solver.runhistory
        trajectory = self.smac.solver.intensifier.traj_logger.trajectory
        self.incumbent = self.smac.solver.incumbent

        # Fetch the results.
        runkeys = list(runhistory.data.keys())
        for key in runkeys:
            reward = 1 - runhistory.data[key][0]
            self.configs_list.append(runhistory.ids_config[key[0]])
            self.config_values.append(reward)

        # Record the time cost.
        time_point = time.time() - self.start_time
        tmp_list = list()
        tmp_list.append(time_point)
        for key in reversed(runkeys[1:]):
            time_point -= runhistory.data[key][1]
            tmp_list.append(time_point)
        self.timing_list.extend(reversed(tmp_list))

        self.logger.info('SMAC smbo ==> the size of evaluations: %d' % len(self.configs_list))
        if len(self.configs_list) > 0:
<<<<<<< HEAD
            self.logger.info('SMAC smbo ==> The time points: %s' % self.timing_list)
=======
            self.logger.info('SMAC smbo ==> The time points: %s' % time_list)
>>>>>>> 8f371cf61c8fb225ebfc75059dc361026e8d96e7
            self.logger.info('SMAC smbo ==> The best performance found: %f' % max(self.config_values))
            self.logger.info('SMAC smbo ==> The best HP found: %s' % self.incumbent)

            # Save the experimental results.
            data = dict()
            data['configs'] = self.configs_list
            data['perfs'] = self.config_values
<<<<<<< HEAD
            data['time_cost'] = self.timing_list
=======
            data['time_cost'] = time_list
>>>>>>> 8f371cf61c8fb225ebfc75059dc361026e8d96e7
            dataset_id = self.result_file.split('_')[0]
            with open('data/%s/' % dataset_id + self.result_file, 'wb') as f:
                pickle.dump(data, f)

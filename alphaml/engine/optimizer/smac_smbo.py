import time
import pickle
import numpy as np
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from alphaml.engine.optimizer.base_optimizer import BaseOptimizer


class SMAC_SMBO(BaseOptimizer):
    def __init__(self, evaluator, config_space, data, seed, **kwargs):
        super().__init__(evaluator, config_space, data, kwargs['metric'], seed)
        self.result_file = kwargs['task_name'] if 'task_name' in kwargs else 'default'
        self.result_file += '_smac.data'

        # Scenario object
        scenario_dict = {
            'abort_on_first_run_crash': False,
            "run_obj": "quality",
            "cs": self.config_space,
            "deterministic": "true"
        }
        if 'runcount' in kwargs and kwargs['runcount'] > 0:
            scenario_dict['runcount-limit'] = kwargs['runcount']
        self.scenario = Scenario(scenario_dict)
        self.smac = SMAC(scenario=self.scenario, rng=np.random.RandomState(self.seed), tae_runner=self.evaluator)

    def run(self):
        configs_list = list()
        config_values = list()
        time_list = list()
        start_time = time.time()

        self.smac.optimize()
        runhistory = self.smac.solver.runhistory
        trajectory = self.smac.solver.intensifier.traj_logger.trajectory
        self.incumbent = self.smac.solver.incumbent

        # Fetch the results.
        runkeys = list(runhistory.data.keys())
        for key in runkeys:
            reward = 1 - runhistory.data[key][0]
            configs_list.append(runhistory.ids_config[key[0]])
            config_values.append(reward)

        # Record the time cost.
        time_point = time.time() - start_time
        tmp_list = list()
        tmp_list.append(time_point)
        for key in reversed(runkeys[1:]):
            time_point -= runhistory.data[key][1]
            tmp_list.append(time_point)
        time_list.extend(reversed(tmp_list))

        self.logger.info('SMAC smbo ==> the size of evaluations: %d' % len(configs_list))
        if len(configs_list) > 0:
            self.logger.info('SMAC smbo ==> The time points: %s' % time_list)
            self.logger.info('SMAC smbo ==> The best performance found: %f' % max(config_values))
            self.logger.info('SMAC smbo ==> The best HP found: %s' % self.incumbent)

            # Save the experimental results.
            data = dict()
            data['configs'] = configs_list
            data['perfs'] = config_values
            data['time_cost'] = time_list
            with open('data/' + self.result_file, 'wb') as f:
                pickle.dump(data, f)

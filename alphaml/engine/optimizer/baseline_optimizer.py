import time
import pickle
import numpy as np
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from litesmac.scenario.scenario import Scenario
from litesmac.facade.smac_facade import SMAC
from alphaml.engine.optimizer.base_optimizer import BaseOptimizer
from alphaml.engine.components.models.classification import _classifiers


class BASELINE(BaseOptimizer):
    def __init__(self, evaluator, config_space, data, seed, **kwargs):
        super().__init__(evaluator, config_space, data, kwargs['metric'], seed)

        self.iter_num = int(1e10) if ('runcount' not in kwargs or kwargs['runcount'] is None) else kwargs['runcount']
        self.estimator_arms = self.config_space.get_hyperparameter('estimator').choices
        self.task_name = kwargs['task_name'] if 'task_name' in kwargs else 'default'
        # update_mode = 1: the random search of an algorithm.
        # update_mode = 2: assign each algorthm with T/N budgets.
        self.update_mode = kwargs['update_mode'] if 'update_mode' in kwargs else 1
        self.result_file = self.task_name + '_%s_smac.data' % ('rand' if self.update_mode == 1 else 'avg')
        self.smac_containers = dict()
        self.rewards, self.cnts = dict(), dict()
        self.configs_list = list()
        self.config_values = list()
        self.time_list = list()

        for estimator in self.estimator_arms:
            # Scenario object
            config_space = _classifiers[estimator].get_hyperparameter_search_space()
            estimator_hp = CategoricalHyperparameter("estimator", [estimator], default_value=estimator)
            config_space.add_hyperparameter(estimator_hp)
            scenario_dict = {
                'abort_on_first_run_crash': False,
                "run_obj": "quality",
                "cs": config_space,
                "deterministic": "true"
            }

            smac = SMAC(scenario=Scenario(scenario_dict),
                        rng=np.random.RandomState(self.seed), tae_runner=self.evaluator)
            self.smac_containers[estimator] = smac
            self.rewards[estimator] = list()
            self.cnts[estimator] = 0

    def run_block(self, B, arm):
        best_arm = arm
        self.logger.info('Choosing to optimize %s arm' % best_arm)
        iter_num, run_cnt = 0, 0

        while True:
            start_time = time.time()
            self.smac_containers[best_arm].iterate()
            runhistory = self.smac_containers[best_arm].solver.runhistory

            # Observe the reward.
            runkeys = list(runhistory.data.keys())
            for key in runkeys[self.cnts[best_arm]:]:
                reward = 1 - runhistory.data[key][0]
                self.rewards[best_arm].append(reward)
                self.configs_list.append(runhistory.ids_config[key[0]])
                self.config_values.append(reward)

            # Record the time cost.
            time_point = time.time() - start_time
            tmp_list = list()
            tmp_list.append(time_point)
            for key in reversed(runkeys[self.cnts[best_arm] + 1:]):
                time_point -= runhistory.data[key][1]
                tmp_list.append(time_point)
            self.time_list.extend(reversed(tmp_list))

            self.logger.info('Iteration %d, the best reward found is %f' % (iter_num, max(self.config_values)))
            iter_num += (len(runkeys) - self.cnts[best_arm])
            self.cnts[best_arm] = len(runhistory.data.keys())
            run_cnt = run_cnt + (2 if run_cnt != 0 else 3)

            if iter_num >= B or run_cnt >= B or (iter_num == 1 and run_cnt > 1):
                break

    def run(self):

        self.logger.info('Start task: %s' % self.task_name)

        if self.update_mode == 1:
            sampled_arm = np.random.choice(self.estimator_arms, 1)[0]
            self.logger.info('Start to optimize %s with B=%d' % (sampled_arm, self.iter_num))
            self.run_block(self.iter_num, sampled_arm)
        elif self.update_mode == 2:
            arm_num = len(self.estimator_arms)
            K = self.iter_num // arm_num
            # Shuffle the arms.
            arm_list = list(self.estimator_arms)
            np.random.shuffle(arm_list)
            for i, arm in enumerate(arm_list):
                B = K
                if i == len(self.estimator_arms) - 1:
                    B = K + self.iter_num % arm_num
                self.logger.info('Start to optimize %s with B=%d' % (arm, B))
                self.run_block(B, arm)
        else:
            raise ValueError('Invalid update mode: %d' % self.update_mode)

        # Print the tuning result.
        self.logger.info('==> the size of evaluations: %d' % len(self.configs_list))
        if len(self.configs_list) > 0:
            id = np.argmax(self.config_values)
            self.logger.info('==> The time points: %s' % self.time_list)
            self.logger.info('==> The best performance found: %f' % self.config_values[id])
            self.logger.info('==> The best HP found: %s' % self.configs_list[id])
            self.incumbent = self.configs_list[id]

            # Save the experimental results.
            data = dict()
            data['configs'] = self.configs_list
            data['perfs'] = self.config_values
            data['time_cost'] = self.time_list
            dataset_id = self.result_file.split('_')[0]
            with open('data/%s/' % dataset_id + self.result_file, 'wb') as f:
                pickle.dump(data, f)

import math
import time
import pickle
import numpy as np
from scipy.stats import norm
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from litesmac.scenario.scenario import Scenario
from litesmac.facade.smac_facade import SMAC
from alphaml.engine.optimizer.base_optimizer import BaseOptimizer
from alphaml.engine.components.models.classification import _classifiers


class TS_NON_SMBO(BaseOptimizer):
    def __init__(self, evaluator, config_space, data, seed, **kwargs):
        super().__init__(evaluator, config_space, data, kwargs['metric'], seed)

        self.iter_num = int(1e10) if ('runcount' not in kwargs or kwargs['runcount'] is None) else kwargs['runcount']
        self.estimator_arms = self.config_space.get_hyperparameter('estimator').choices
        self.task_name = kwargs['task_name'] if 'task_name' in kwargs else 'default'
        self.result_file = self.task_name + '_ts_non_smac.data'
        self.update_mode = kwargs['update_mode'] if 'update_mode' in kwargs else 1
        self.smac_containers = dict()
        self.ts_cnts = dict()
        self.ts_rewards = dict()
        self.weight = None

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
            self.ts_cnts[estimator] = 0
            self.ts_rewards[estimator] = list()

    def run(self):
        configs_list = list()
        config_values = list()
        time_list = list()
        iter_num = 0
        best_perf = 0.
        start_time = time.time()
        self.logger.info('Start task: %s' % self.task_name)

        K = len(self.estimator_arms)
        gamma = 0.3
        delta_t = 50
        iter_id = 0

        while True:
            if iter_id % delta_t == 0:
                self.weight = np.ones(K)
            iter_id += 1
            # Obtain the p vector.
            p = (1 - gamma) * self.weight/np.sum(self.weight) + gamma/K
            # Draw an arm.
            best_index = np.random.choice(K, 1, p=p)[0]
            best_arm = self.estimator_arms[best_index]
            self.logger.info('Choosing to optimize %s arm' % best_arm)

            self.smac_containers[best_arm].iterate()
            runhistory = self.smac_containers[best_arm].solver.runhistory

            # Observe the reward.
            update_flag = False
            best_reward = max(self.ts_rewards[best_arm]) if len(self.ts_rewards[best_arm]) > 0 else 0
            runkeys = list(runhistory.data.keys())
            for key in runkeys[self.ts_cnts[best_arm]:]:
                reward = 1 - runhistory.data[key][0]
                best_reward = reward if reward > best_reward else best_reward
                if reward >= best_perf:
                    update_flag = True
                    best_perf = reward
                self.ts_rewards[best_arm].append(reward)
                configs_list.append(runhistory.ids_config[key[0]])
                config_values.append(reward)

            # Record the time cost.
            time_point = time.time() - start_time
            tmp_list = list()
            tmp_list.append(time_point)
            for key in reversed(runkeys[self.ts_cnts[best_arm]+1:]):
                time_point -= runhistory.data[key][1]
                tmp_list.append(time_point)
            time_list.extend(reversed(tmp_list))

            self.logger.info('Iteration %d, the best reward found is %f' % (iter_num, max(config_values)))
            iter_num += (len(runkeys) - self.ts_cnts[best_arm])
            self.ts_cnts[best_arm] = len(runhistory.data.keys())

            # Update the weight w vector.
            if self.update_mode == 1:
                x_bar = best_reward/p[best_index]
                self.weight[best_index] *= np.exp(gamma*x_bar/K)
            else:
                raise ValueError('Invalid update mode: %d' % self.update_mode)

            if iter_num >= self.iter_num:
                break

            # Print the parameters in Thompson sampling.
            self.logger.info('Vector p: %s' % dict(zip(self.estimator_arms, p)))

        # Print the parameters in Thompson sampling.
        self.logger.info('ts params: %s' % self.weight)
        self.logger.info('ts counts: %s' % self.ts_cnts)
        self.logger.info('ts rewards: %s' % self.ts_rewards)

        # Print the tuning result.
        self.logger.info('TS smbo ==> the size of evaluations: %d' % len(configs_list))
        if len(configs_list) > 0:
            id = np.argmax(config_values)
            self.logger.info('TS smbo ==> The time points: %s' % time_list)
            self.logger.info('TS smbo ==> The best performance found: %f' % config_values[id])
            self.logger.info('TS smbo ==> The best HP found: %s' % configs_list[id])
            self.incumbent = configs_list[id]

            # Save the experimental results.
            data = dict()
            data['ts_weight'] = self.weight
            data['ts_cnts'] = self.ts_cnts
            data['ts_rewards'] = self.ts_rewards
            data['configs'] = configs_list
            data['perfs'] = config_values
            data['time_cost'] = time_list
            dataset_id = self.result_file.split('_')[0]
            with open('data/%s/' % dataset_id + self.result_file, 'wb') as f:
                pickle.dump(data, f)

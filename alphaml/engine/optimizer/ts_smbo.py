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


class TS_SMBO(BaseOptimizer):
    def __init__(self, evaluator, config_space, data, seed, **kwargs):
        super().__init__(evaluator, config_space, data, kwargs['metric'], seed)

        self.iter_num = int(1e10) if ('runcount' not in kwargs or kwargs['runcount'] is None) else kwargs['runcount']
        self.estimator_arms = self.config_space.get_hyperparameter('estimator').choices
        self.task_name = kwargs['task_name'] if 'task_name' in kwargs else 'default'
        self.result_file = self.task_name + '_ts_smac.data'

        self.update_mode = 1
        self.smac_containers = dict()
        self.ts_params = dict()
        self.ts_cnts = dict()
        self.penalty_factor = dict()
        self.gamma = 0.97
        self.ts_rewards = dict()
        self.alphas = dict()

        self.max_dim = 0
        for estimator in self.estimator_arms:
            # Scenario object
            num_dim = len(_classifiers[estimator].get_hyperparameter_search_space().get_hyperparameters())
            if num_dim > self.max_dim:
                self.max_dim = num_dim

        for estimator in self.estimator_arms:
            # Scenario object
            config_space = _classifiers[estimator].get_hyperparameter_search_space()
            num_dim = len(config_space.get_hyperparameters())
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
            self.ts_params[estimator] = [0.5, 1.]
            self.ts_cnts[estimator] = 0
            self.penalty_factor[estimator] = 1.
            self.ts_rewards[estimator] = list()
            if num_dim == 0:
                self.alphas[estimator] = 1e10
            else:
                self.alphas[estimator] = math.sqrt(self.max_dim/num_dim)

    def run(self):
        configs_list = list()
        config_values = list()
        time_list = list()
        iter_num = 0
        best_perf = 0.
        start_time = time.time()
        self.logger.info('Start task: %s' % self.task_name)

        while True:
            samples = list()
            for estimator in self.estimator_arms:
                sample = norm.rvs(loc=self.ts_params[estimator][0], scale=self.ts_params[estimator][1])
                samples.append(sample)
            best_arm = self.estimator_arms[np.argmax(samples)]
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

            # Update the posterior estimation.
            # TODO: rethinking.
            if self.update_mode == 1:
                self.ts_params[best_arm][0] = best_reward
                if update_flag:
                    # if the update is the best, penalty gives to other arms.
                    check_flag = [True for est in self.estimator_arms if self.ts_cnts[est] >= 3]
                    if np.all(check_flag):
                        for est in self.estimator_arms:
                            if est != best_arm:
                                self.penalty_factor[est] *= self.gamma
                        print('Penalty factor', self.penalty_factor)

                self.ts_params[best_arm][1] = self.penalty_factor[best_arm] * 0.1667 / \
                                              (self.alphas[best_arm] * self.ts_cnts[best_arm] + 1)
            else:
                raise ValueError('Invalid update mode: %d' % self.update_mode)
            if iter_num >= self.iter_num:
                break

            # Print the parameters in Thompson sampling.
            self.logger.info('ts params: %s' % self.ts_params)

        # Print the parameters in Thompson sampling.
        self.logger.info('ts params: %s' % self.ts_params)
        self.logger.info('ts counts: %s' % self.ts_cnts)
        self.logger.info('ts rewards: %s' % self.ts_rewards)
        self.logger.info('ts penalty: %s' % self.penalty_factor)

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
            data['ts_params'] = self.ts_params
            data['ts_cnts'] = self.ts_cnts
            data['ts_rewards'] = self.ts_rewards
            data['ts_penalty'] = self.penalty_factor
            data['configs'] = configs_list
            data['perfs'] = config_values
            data['time_cost'] = time_list
            dataset_id = self.result_file.split('_')[0]
            with open('data/%s/' % dataset_id + self.result_file, 'wb') as f:
                pickle.dump(data, f)

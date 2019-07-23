import math
import time
import pickle
import numpy as np
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from litesmac.scenario.scenario import Scenario
from litesmac.facade.smac_facade import SMAC
from alphaml.engine.optimizer.base_optimizer import BaseOptimizer
from alphaml.engine.components.models.classification import _classifiers


class UCB_SMBO(BaseOptimizer):
    def __init__(self, evaluator, config_space, data, seed, **kwargs):
        super().__init__(evaluator, config_space, data, kwargs['metric'], seed)

        self.iter_num = int(1e10) if ('runcount' not in kwargs or kwargs['runcount'] is None) else kwargs['runcount']
        self.estimator_arms = self.config_space.get_hyperparameter('estimator').choices
        self.task_name = kwargs['task_name'] if 'task_name' in kwargs else 'default'

        self.mode = kwargs['update_mode'] if 'update_mode' in kwargs else 1
        self.param = float(kwargs['param']) if 'param' in kwargs else None

        self.result_file = self.task_name + '_ucb_mab_%d_smac.data' % self.mode
        self.logger.info('Result file: %s' % self.result_file)
        self.smac_containers = dict()
        self.ts_cnts = dict()
        self.ts_rewards = dict()
        self.weight = None
        self.lbd = 0.9
        self.alphas = dict()

        self.configs_list = list()
        self.config_values = list()

        self.max_dim = 0
        for estimator in self.estimator_arms:
            # Scenario object
            num_dim = len(_classifiers[estimator].get_hyperparameter_search_space().get_hyperparameters())
            if num_dim > self.max_dim:
                self.max_dim = num_dim

        for estimator in self.estimator_arms:
            # Scenario object
            config_space = _classifiers[estimator].get_hyperparameter_search_space()
            estimator_hp = CategoricalHyperparameter("estimator", [estimator], default_value=estimator)
            num_dim = len(config_space.get_hyperparameters())
            config_space.add_hyperparameter(estimator_hp)
            scenario_dict = {
                'abort_on_first_run_crash': False,
                "run_obj": "quality",
                "cs": config_space,
                "deterministic": "true"
            }

            smac = SMAC(scenario=Scenario(scenario_dict), rng=np.random.RandomState(self.seed),
                        tae_runner=self.evaluator)
            self.smac_containers[estimator] = smac
            self.ts_cnts[estimator] = 0
            self.ts_rewards[estimator] = list()
            self.alphas[estimator] = 0. if num_dim == 0 else math.sqrt(self.max_dim/num_dim)

    def run(self):
        time_list = list()
        iter_num = 0
        start_time = time.time()
        self.logger.info('Start task: %s' % self.task_name)

        K = len(self.estimator_arms)
        iter_id = 0
        self.weight = np.zeros(K)

        while True:
            iter_id += 1
            if iter_id <= K:
                # First iterate each algorithm.
                best_index = iter_id - 1
                best_arm = self.estimator_arms[best_index]
            else:
                if self.mode == 1:
                    confidence_value = np.array([np.sqrt(self.alphas[est] * np.log(iter_num + 1) / self.ts_cnts[est])
                                                 for est in self.estimator_arms])
                    q_value = self.weight + confidence_value
                    self.logger.info('Q vector: %s' % q_value)
                    best_index = np.argmax(q_value)
                    best_arm = self.estimator_arms[best_index]
                elif self.mode == 2:
                    lbd = 1.
                    p_min = 0.01
                    exp_sum = np.sum(np.exp(self.weight/lbd))
                    p = p_min + (1 - K*p_min) * np.exp(self.weight/lbd) / exp_sum
                    self.logger.info('P vector: %s' % p)
                    # Draw an arm.
                    best_index = np.random.choice(K, 1, p=p)[0]
                    best_arm = self.estimator_arms[best_index]
            self.logger.info('Choosing to optimize %s arm' % best_arm)

            self.smac_containers[best_arm].iterate()
            runhistory = self.smac_containers[best_arm].solver.runhistory

            # Observe the reward.
            runkeys = list(runhistory.data.keys())
            for key in runkeys[self.ts_cnts[best_arm]:]:
                reward = 1 - runhistory.data[key][0]
                self.ts_rewards[best_arm].append(reward)
                self.configs_list.append(runhistory.ids_config[key[0]])
                self.config_values.append(reward)

            # Record the time cost.
            time_point = time.time() - start_time
            tmp_list = list()
            tmp_list.append(time_point)
            for key in reversed(runkeys[self.ts_cnts[best_arm]+1:]):
                time_point -= runhistory.data[key][1]
                tmp_list.append(time_point)
            time_list.extend(reversed(tmp_list))

            self.logger.info('Iteration %d, the best reward found is %f' % (iter_num, max(self.config_values)))
            iter_num += (len(runkeys) - self.ts_cnts[best_arm])
            self.ts_cnts[best_arm] = len(runhistory.data.keys())

            # Update the weight w vector.
            self.weight[best_index] = (1 - self.lbd)*self.weight[best_index] + self.lbd * max(self.ts_rewards[best_arm])

            if iter_num >= self.iter_num:
                break

            # Print the parameters in Thompson sampling.
            self.logger.info('Vector Weight: %s' % dict(zip(self.estimator_arms, self.weight)))

        # Print the parameters in Thompson sampling.
        self.logger.info('ts params: %s' % self.weight)
        self.logger.info('ts counts: %s' % self.ts_cnts)
        self.logger.info('ts rewards: %s' % self.ts_rewards)

        # Print the tuning result.
        self.logger.info('non-mab ==> the size of evaluations: %d' % len(self.configs_list))
        if len(self.configs_list) > 0:
            id = np.argmax(self.config_values)
            self.logger.info('non-mab ==> The time points: %s' % time_list)
            self.logger.info('non-mab ==> The best performance found: %f' % self.config_values[id])
            self.logger.info('non-mab ==> The best HP found: %s' % self.configs_list[id])
            self.incumbent = self.configs_list[id]

            # Save the experimental results.
            data = dict()
            data['ts_weight'] = self.weight
            data['ts_cnts'] = self.ts_cnts
            data['ts_rewards'] = self.ts_rewards
            data['configs'] = self.configs_list
            data['perfs'] = self.config_values
            data['time_cost'] = time_list
            dataset_id = self.result_file.split('_')[0]
            with open('data/%s/' % dataset_id + self.result_file, 'wb') as f:
                pickle.dump(data, f)

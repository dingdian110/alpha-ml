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


class MONO_MAB_SMBO(BaseOptimizer):
    def __init__(self, evaluator, config_space, data, seed, **kwargs):
        super().__init__(evaluator, config_space, data, kwargs['metric'], seed)

        self.iter_num = int(1e10) if ('runcount' not in kwargs or kwargs['runcount'] is None) else kwargs['runcount']
        self.estimator_arms = self.config_space.get_hyperparameter('estimator').choices
        self.mode = kwargs['update_mode'] if 'update_mode' in kwargs else 1
        self.task_name = kwargs['task_name'] if 'task_name' in kwargs else 'default'
        self.result_file = self.task_name + '_ts_mm_%d_smac.data' % self.mode

        self.smac_containers = dict()
        self.ts_cnts = dict()
        self.ts_rewards = dict()
        self.updated_rewards = dict()

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
            self.updated_rewards[estimator] = list()

    def run(self):
        configs_list = list()
        config_values = list()
        time_list = list()
        iter_num = 0
        start_time = time.time()
        self.logger.info('Start task: %s' % self.task_name)

        arm_set = list(self.estimator_arms)
        tmp_iter = 0
        T = self.iter_num
        duration = 10

        while True:
            # Pull each arm exactly once.
            tmp_iter += 1
            p, q = list(), list()

            for arm in arm_set:
                self.logger.info('Choosing to optimize %s arm' % arm)
                self.smac_containers[arm].iterate()
                runhistory = self.smac_containers[arm].solver.runhistory

                # Observe the reward.
                runkeys = list(runhistory.data.keys())
                for key in runkeys[self.ts_cnts[arm]:]:
                    reward = 1 - runhistory.data[key][0]
                    self.ts_rewards[arm].append(reward)
                    self.updated_rewards[arm].append(max(self.ts_rewards[arm]))
                    configs_list.append(runhistory.ids_config[key[0]])
                    config_values.append(reward)

                # Record the time cost.
                time_point = time.time() - start_time
                tmp_list = list()
                tmp_list.append(time_point)
                for key in reversed(runkeys[self.ts_cnts[arm]+1:]):
                    time_point -= runhistory.data[key][1]
                    tmp_list.append(time_point)
                time_list.extend(reversed(tmp_list))

                iter_num += (len(runkeys) - self.ts_cnts[arm])
                self.ts_cnts[arm] = len(runhistory.data.keys())

                acc_reward = self.updated_rewards[arm]
                if self.ts_cnts[arm] > 2:
                    if len(acc_reward) >= duration:
                        estimated_slope = (acc_reward[-1] - acc_reward[-duration]) / duration
                    else:
                        estimated_slope = (acc_reward[-1] - acc_reward[0]) / len(acc_reward)

                    if self.mode == 1:
                        F = sum(acc_reward)
                        pred = sum([min(1., acc_reward[-1] + estimated_slope * (t - tmp_iter))
                                    for t in range(tmp_iter+1, T)])
                        p.append(F + pred)
                        q.append(F + acc_reward[-1]*(T - tmp_iter))
                    elif self.mode == 2:
                        p.append(min(1., acc_reward[-1] + estimated_slope * (T - tmp_iter)))
                        q.append(acc_reward[-1])
                    else:
                        raise ValueError('Invalid mode: %d.' % self.mode)
                    print('Slope, P, Q: %.4f' % estimated_slope)
                    print(arm_set, p, q)
                else:
                    p.append(acc_reward[-1])
                    q.append(acc_reward[-1])

            self.logger.info('Iteration %d, the best reward found is %f' % (iter_num, max(config_values)))

            # Remove some arm.
            N = len(arm_set)
            flags = [False] * N

            for i in range(N):
                for j in range(N):
                    if i != j:
                        if p[i] < q[j]:
                            flags[i] = True

            self.logger.info('>>>>> Remove Models: %s' % [item for index, item in enumerate(arm_set) if flags[index]])
            arm_set = [item for index, item in enumerate(arm_set) if not flags[index]]

            if iter_num >= self.iter_num:
                break

        # Print the parameters in Thompson sampling.
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
            data['ts_cnts'] = self.ts_cnts
            data['ts_rewards'] = self.ts_rewards
            data['configs'] = configs_list
            data['perfs'] = config_values
            data['time_cost'] = time_list
            dataset_id = self.result_file.split('_')[0]
            with open('data/%s/' % dataset_id + self.result_file, 'wb') as f:
                pickle.dump(data, f)

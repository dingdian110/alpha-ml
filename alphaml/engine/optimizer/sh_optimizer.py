import time
import pickle
import numpy as np
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from litesmac.scenario.scenario import Scenario
from litesmac.facade.smac_facade import SMAC
from alphaml.engine.optimizer.base_optimizer import BaseOptimizer
from alphaml.engine.components.models.classification import _classifiers


class SH_SMBO(BaseOptimizer):
    def __init__(self, evaluator, config_space, data, seed, **kwargs):
        super().__init__(evaluator, config_space, data, kwargs['metric'], seed)

        self.iter_num = int(1e10) if ('runcount' not in kwargs or kwargs['runcount'] is None) else kwargs['runcount']
        self.estimator_arms = self.config_space.get_hyperparameter('estimator').choices
        self.task_name = kwargs['task_name'] if 'task_name' in kwargs else 'default'
        self.eta = 2
        self.proportion = kwargs['param'] if 'param' in kwargs else 1.
        self.result_file = self.task_name + '_sh_smac_%.2f.data' % self.proportion

        self.smac_containers = dict()
        self.cnts = dict()
        self.rewards = dict()
        self.configs_list = list()
        self.config_values = list()

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
            self.cnts[estimator] = 0
            self.rewards[estimator] = list()

    def run(self):
        self.logger.info('Start task: %s' % self.task_name)
        n_arms = len(self.estimator_arms)
        arm_set = list(self.estimator_arms)
        iter_num = 0
        k = 0
        # The budget used to identify the best arm.
        B = int(self.iter_num * self.proportion)

        while True:
            n_k = len(arm_set)
            r_k = int(np.floor(B/(n_k*np.ceil(np.log2(n_arms)))))
            r_k = max(1, r_k//2)
            if n_k == 1:
                r_k = (self.iter_num - iter_num)//2 + 1

            self.logger.info('Iteration %s: r_k = %d, n_k = %d' % (k, r_k, n_k))

            perf = list()
            es_flag = False

            # Pull each arm with r_k units of resource.
            for arm in arm_set:
                self.logger.info('Optimize arm %s with %d units of budget!' % (arm, r_k))
                for _ in range(r_k):
                    self.smac_containers[arm].iterate()
                runhistory = self.smac_containers[arm].solver.runhistory

                # Observe the reward.
                runkeys = list(runhistory.data.keys())
                for key in runkeys[self.cnts[arm]:]:
                    reward = 1 - runhistory.data[key][0]
                    self.rewards[arm].append(reward)
                    self.configs_list.append(runhistory.ids_config[key[0]])
                    self.config_values.append(reward)

                # Determine whether to stop early.
                if len(arm_set) == 1 and len(runkeys[self.cnts[arm]:]) == 0:
                    es_flag = True

                # Record the time cost.
                time_point = time.time() - self.start_time
                tmp_list = list()
                tmp_list.append(time_point)
                for key in reversed(runkeys[self.cnts[arm]+1:]):
                    time_point -= runhistory.data[key][1]
                    tmp_list.append(time_point)
                self.timing_list.extend(reversed(tmp_list))

                iter_num += (len(runkeys) - self.cnts[arm])
                self.cnts[arm] = len(runhistory.data.keys())
                perf.append(max(self.rewards[arm]))
                self.logger.info('Iteration %d, the best reward found >>> %f!' % (iter_num, max(self.config_values)))

            if n_k > 1:
                indices = np.argsort(perf)[int(np.ceil(n_k/self.eta)):]
                arm_set = [item for index, item in enumerate(arm_set) if index in indices]
                self.logger.info('Arms left: %s' % arm_set)
            k += 1
            if iter_num >= self.iter_num or es_flag:
                break

        # Print the parameters in Thompson sampling.
        self.logger.info('SH counts: %s' % self.cnts)
        self.logger.info('SH rewards: %s' % self.rewards)

        # Print the tuning result.
        self.logger.info('SH smbo ==> the size of evaluations: %d' % len(self.configs_list))
        if len(self.configs_list) > 0:
            id = np.argmax(self.config_values)
            self.logger.info('SH smbo ==> The time points: %s' % self.timing_list)
            self.logger.info('SH smbo ==> The best performance found: %f' % self.config_values[id])
            self.logger.info('SH smbo ==> The best HP found: %s' % self.configs_list[id])
            self.incumbent = self.configs_list[id]

            # Save the experimental results.
            data = dict()
            data['ts_cnts'] = self.cnts
            data['ts_rewards'] = self.rewards
            data['configs'] = self.configs_list
            data['perfs'] = self.config_values
            data['time_cost'] = self.timing_list
            dataset_id = self.result_file.split('_')[0]
            with open('data/%s/' % dataset_id + self.result_file, 'wb') as f:
                pickle.dump(data, f)

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
        self.result_file = self.task_name + '_sh_smac.data'

        self.smac_containers = dict()
        self.cnts = dict()
        self.rewards = dict()

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
        configs_list = list()
        config_values = list()
        time_list = list()
        iter_num = 0
        start_time = time.time()
        self.logger.info('Start task: %s' % self.task_name)

        arm_set = list(self.estimator_arms)
        B = self.iter_num
        n = len(self.estimator_arms)
        K = int(np.ceil(np.log2(n))) - 1
        S = n
        k = 0
        while True:
            r_k = np.floor(B/(S*np.ceil(np.log2(n))))
            self.logger.info('Iteration %s: r_k = %d, S_k = %d' % (k, r_k, S))
            smbo_cnt = int(r_k) // 2
            if S == 1:
                smbo_cnt = (B - iter_num)//2 + 1
            perf = list()
            # Pull each arm with r_k units of resource.
            for arm in arm_set:
                self.logger.info('Optimize arm %s with %d budget' % (arm, smbo_cnt))
                for _ in range(smbo_cnt):
                    self.smac_containers[arm].iterate()
                runhistory = self.smac_containers[arm].solver.runhistory

                # Observe the reward.
                runkeys = list(runhistory.data.keys())
                for key in runkeys[self.cnts[arm]:]:
                    reward = 1 - runhistory.data[key][0]
                    self.rewards[arm].append(reward)
                    configs_list.append(runhistory.ids_config[key[0]])
                    config_values.append(reward)

                # Record the time cost.
                time_point = time.time() - start_time
                tmp_list = list()
                tmp_list.append(time_point)
                for key in reversed(runkeys[self.cnts[arm]+1:]):
                    time_point -= runhistory.data[key][1]
                    tmp_list.append(time_point)
                time_list.extend(reversed(tmp_list))

                iter_num += (len(runkeys) - self.cnts[arm])
                self.cnts[arm] = len(runhistory.data.keys())
                perf.append(max(self.rewards[arm]))

            self.logger.info('Iteration %d, the best reward found is %f' % (iter_num, max(config_values)))
            if S > 1:
                indices = np.argsort(perf)[S//2:]
                arm_set = [item for index, item in enumerate(arm_set) if index in indices]
                self.logger.info('Left arms: %s' % arm_set)
                S = len(arm_set)
            k += 1
            if iter_num >= self.iter_num:
                break

        # Print the parameters in Thompson sampling.
        self.logger.info('ts counts: %s' % self.cnts)
        self.logger.info('ts rewards: %s' % self.rewards)

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
            data['ts_cnts'] = self.cnts
            data['ts_rewards'] = self.rewards
            data['configs'] = configs_list
            data['perfs'] = config_values
            data['time_cost'] = time_list
            dataset_id = self.result_file.split('_')[0]
            with open('data/%s/' % dataset_id + self.result_file, 'wb') as f:
                pickle.dump(data, f)

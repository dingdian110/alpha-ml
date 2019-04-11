import numpy as np
from scipy.stats import norm
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from litesmac.scenario.scenario import Scenario
from litesmac.facade.smac_facade import SMAC
from alphaml.engine.optimizer.base_optimizer import BaseOptimizer
from alphaml.engine.components.models.classification import _classifiers


class TS_SMBO(BaseOptimizer):
    def __init__(self, evaluator, config_space, data, metric, seed):
        super().__init__(evaluator, config_space, data, metric, seed)

        self.iter_num = 25
        self.estimator_arms = self.config_space.get_hyperparameter('estimator').choices

        self.smac_containers = dict()
        self.ts_params = dict()
        self.ts_cnts = dict()
        self.ts_rewards = dict()

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
            self.ts_params[estimator] = [0, 1]
            self.ts_cnts[estimator] = 0
            self.ts_rewards[estimator] = list()

    def run(self):
        incubent_values = list()

        for _ in range(self.iter_num):
            samples = list()
            for estimator in self.estimator_arms:
                sample = norm.rvs(loc=self.ts_params[estimator][0], scale=self.ts_params[estimator][1])
                samples.append(sample)
            best_arm = self.estimator_arms[np.argmax(samples)]

            self.smac_containers[best_arm].iterate()
            runhistory = self.smac_containers[best_arm].solver.runhistory

            # Observe the reward.
            reward = 1 - runhistory.get_cost(runhistory.get_all_configs()[-1])
            incubent_values.append(reward)
            self.ts_rewards[estimator].append(reward)
            self.ts_cnts[estimator] += 1

            # Update the posterior estimation.
            self.ts_params[estimator][0] = np.mean(self.ts_rewards[estimator])
            self.ts_params[estimator][1] = 1./(self.ts_params[estimator][1] + 1)

        perfs = list()
        incumbent, inc_val = None, np.inf
        for arm in self.estimator_arms:
            runhistory = self.smac_containers[arm].solver.runhistory
            inc = self.smac_containers[arm].solver.incumbent
            if runhistory.get_cost(inc) < inc_val:
                incumbent, inc_val = inc, runhistory.get_cost(inc)

            configs = runhistory.get_all_configs()
            for config in configs:
                perfs.append(runhistory.get_cost(config))
        print(len(perfs))
        print(perfs)
        print(min(perfs), max(perfs))
        self.incumbent = incumbent

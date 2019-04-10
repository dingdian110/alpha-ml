import numpy as np
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from alphaml.engine.optimizer.base_optimizer import BaseOptimizer
from alphaml.engine.evaluator.base import BaseEvaluator


class SMAC_SMBO(BaseOptimizer):
    def __init__(self, evaluator, config_space, data, metric, seed):
        super().__init__(evaluator, config_space, data, metric, seed)

        # Scenario object
        scenario_dict = {
            'abort_on_first_run_crash': False,
            "run_obj": "quality",
            "cs": self.config_space,
            "deterministic": "true",
            "runcount-limit": 53
        }
        self.scenario = Scenario(scenario_dict)
        self.smac = SMAC(scenario=self.scenario, rng=np.random.RandomState(self.seed), tae_runner=self.evaluator)

    def run(self):
        self.smac.optimize()
        self.runhistory = self.smac.solver.runhistory
        self.trajectory = self.smac.solver.intensifier.traj_logger.trajectory
        self.incumbent = self.smac.solver.incumbent
        # Fetch the results.
        configs = self.runhistory.get_all_configs()
        perfs = dict()
        for config in configs:
            perfs[config] = self.runhistory.get_cost(config)
        print(perfs.values())
        print(min(perfs.values()))
        print(perfs[self.incumbent])
        print(self.incumbent in perfs)
        return self.runhistory, self.trajectory

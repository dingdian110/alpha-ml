import numpy as np
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from alphaml.engine.optimizer.base_optimizer import BaseOptimizer
from alphaml.engine.evaluator.base import BaseEvaluator


class SMAC_SMBO(BaseOptimizer):
    def __init__(self, config_space, data, metric):
        super().__init__(config_space, data, metric)

        # Create evaluator & assign the required data to it.
        self.evaluator = BaseEvaluator(data, metric)

        # Scenario object
        scenario_dict = {
            'abort_on_first_run_crash': False,
            "run_obj": "quality",
            "cs": self.config_space,
            "deterministic": "true",
            "runcount-limit": 100
        }
        self.scenario = Scenario(scenario_dict)
        self.smac = SMAC(scenario=self.scenario, rng=np.random.RandomState(42), tae_runner=self.evaluator)

    def run(self):
        self.smac.optimize()
        self.runhistory = self.smac.solver.runhistory
        self.trajectory = self.smac.solver.intensifier.traj_logger.trajectory

        # Show the results.
        configs = self.runhistory.get_all_configs()
        perfs = list()
        for config in configs:
            perfs.append(self.runhistory.get_cost(config))
        print(len(perfs))
        print(perfs)
        print(min(perfs), max(perfs))
        return self.runhistory, self.trajectory

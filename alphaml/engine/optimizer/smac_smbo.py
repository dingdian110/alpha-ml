import numpy as np
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from alphaml.engine.optimizer.base_optimizer import BaseOptimizer


class SMAC_SMBO(BaseOptimizer):
    def __init__(self, config_space, evaluator):
        super().__init__(config_space, evaluator)

        # Scenario object
        scenario_dict = {
            'abort_on_first_run_crash': False,
            "run_obj": "quality",
            "cs": self.config_space,
            "deterministic": "true",
            "runcount-limit": 50
        }
        self.scenario = Scenario(scenario_dict)
        self.smac = SMAC(scenario=self.scenario, rng=np.random.RandomState(42), tae_runner=self.evaluator)

    def run(self):
        self.smac.optimize()
        self.runhistory = self.smac.solver.runhistory
        self.trajectory = self.smac.solver.intensifier.traj_logger.trajectory
        return self.runhistory, self.trajectory

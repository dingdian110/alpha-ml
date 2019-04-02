import numpy as np
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from alphaml.engine.optimizer.base_optimizer import BaseOptimizer


class SMAC_SMBO(BaseOptimizer):
    def __init__(self, config_space, evaluator):
        super().__init__(config_space, evaluator)

        # Scenario object
        self.scenario = Scenario({"run_obj": "quality",
                                  "runcount-limit": 200,
                                  "cs": self.config_space,
                                  "deterministic": "true"
                                  })
        self.smac = SMAC(scenario=self.scenario, rng=np.random.RandomState(42), tae_runner=self.evaluator)

    def run(self):
        inc = self.smac.optimize()
        return inc

import numpy as np
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from alphaml.engine.evaluator.base import SimpleEvaluator


class SMAC_SMBO(object):
    def __init__(self, config_space):
        self.config_space = config_space

    def run(self):
        # Scenario object
        scenario = Scenario({"run_obj": "quality",
                                  "runcount-limit": 200,
                                  "cs": self.config_space,
                                  "deterministic": "true"
                                  })
        ta = SimpleEvaluator()
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42), tae_runner=ta)

        inc = smac.optimize()
        print(inc)

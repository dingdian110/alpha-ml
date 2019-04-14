import numpy as np
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from alphaml.engine.optimizer.base_optimizer import BaseOptimizer


class SMAC_SMBO(BaseOptimizer):
    def __init__(self, evaluator, config_space, data, seed, **kwargs):
        super().__init__(evaluator, config_space, data, kwargs['metric'], seed)

        # Scenario object
        scenario_dict = {
            'abort_on_first_run_crash': False,
            "run_obj": "quality",
            "cs": self.config_space,
            "deterministic": "true"
        }
        if 'runcount' in kwargs and kwargs['runcount'] > 0:
            scenario_dict['runcount-limit'] = kwargs['runcount']
        self.scenario = Scenario(scenario_dict)
        self.smac = SMAC(scenario=self.scenario, rng=np.random.RandomState(self.seed), tae_runner=self.evaluator)

    def run(self):
        self.smac.optimize()
        self.runhistory = self.smac.solver.runhistory
        self.trajectory = self.smac.solver.intensifier.traj_logger.trajectory
        self.incumbent = self.smac.solver.incumbent
        # Fetch the results.
        configs = self.runhistory.get_all_configs()
        perfs = list()
        for config in configs:
            perfs.append(self.runhistory.get_cost(config))

        flag = 'SMAC smbo ==> '
        self.logger.info(flag + 'the size of evaluations: %d' % len(perfs))
        if len(perfs) > 0:
            self.logger.info(flag + 'The best performance found: %f' % min(perfs))
            self.logger.info(flag + 'The best HP found: %s' % self.incumbent)

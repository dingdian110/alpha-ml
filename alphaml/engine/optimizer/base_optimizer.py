class BaseOptimizer(object):
    def __init__(self, configspace, evaluator):
        self.config_space = configspace
        self.evaluator = evaluator

    def run(self):
        raise NotImplementedError

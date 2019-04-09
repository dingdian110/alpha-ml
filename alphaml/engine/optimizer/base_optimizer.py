class BaseOptimizer(object):
    def __init__(self, evaluator, config_space, data, metric, seed):
        # Prepare the basics for evaluator.
        self.evaluator = evaluator
        self.evaluator.data_manager = data
        self.evaluator.metric = metric
        self.config_space = config_space
        self.seed = seed

    def run(self):
        raise NotImplementedError

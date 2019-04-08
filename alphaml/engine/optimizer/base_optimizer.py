class BaseOptimizer(object):
    def __init__(self, config_space, data, metric, seed):
        self.config_space = config_space
        self.data = data
        self.metric = metric
        self.seed = seed

    def run(self):
        raise NotImplementedError

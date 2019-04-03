class BaseOptimizer(object):
    def __init__(self, config_space, data, metric):
        self.config_space = config_space
        self.data = data
        self.metric = metric

    def run(self):
        raise NotImplementedError

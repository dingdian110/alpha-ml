from alphaml.engine.evaluator.base import BaseEvaluator


class BaseOptimizer(object):
    def __init__(self, evaluator, config_space, data, metric, seed):
        # Prepare the basics for evaluator.
        assert isinstance(evaluator, BaseEvaluator)
        self.evaluator = evaluator
        self.evaluator.data_manager = data
        self.evaluator.metric_func = metric
        self.config_space = config_space
        self.seed = seed
        self.incumbent = None

    def run(self):
        raise NotImplementedError

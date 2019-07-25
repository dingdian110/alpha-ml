import time
import logging
import numpy as np
from alphaml.engine.evaluator.base import BaseEvaluator
from alphaml.engine.components.data_manager import DataManager
from alphaml.utils.constants import MAX_INT


class Optimizer(object):
    def __init__(self, evaluator, config_instance, data, metric, seed):
        # Prepare the basics for evaluator.
        assert isinstance(evaluator, BaseEvaluator)
        self.evaluator = evaluator
        self.evaluator.data_manager = data
        self.evaluator.metric_func = metric
        self.config_instance = config_instance
        if seed is None:
            seed = np.random.random_integers(MAX_INT)
        self.seed = seed
        self.start_time = time.time()
        self.timing_list = list()
        self.incumbent = None
        self.logger = logging.getLogger(__name__)
        self.logger.info('The random seed is: %d' % self.seed)

    """
    @:return return the optimized train and test arrays.
    """
    def optimize(self) -> DataManager:
        return self.evaluator.data_manager

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from alphaml.engine.components.models.classification import _classifiers
from alphaml.utils.constants import *


class ComponentsManager(object):
    def __init__(self):
        self.builtin_cls_models = _classifiers.keys()
        self.builtin_reg_models = []

    def get_hyperparameter_search_space(self, task_type, include=None, exclude=None):
        if task_type != REGRESSION:
            model_candidates = set(self.builtin_cls_models)
            if include is not None:
                for model in include:
                    if model in self.builtin_cls_models:
                        model_candidates.add(model)
                    else:
                        raise ValueError("The estimator %s is NOT available in alpha-ml!")
            if exclude is not None:
                for model in exclude:
                    if model in model_candidates:
                        model_candidates.remove(model)
            return self.get_cls_configuration_space(list(model_candidates))

    def get_cls_configuration_space(self, model_candidates):
        """
        Reference: pipeline/base=325, classification/__init__=121
        """
        cs = ConfigurationSpace()
        # TODO: set the default model.
        model_option = CategoricalHyperparameter("classifier", model_candidates, default_value=model_candidates[0])
        cs.add_hyperparameter(model_option)

        for model_item in model_candidates:
            sub_configuration_space = _classifiers[model_item].get_hyperparameter_search_space()
            parent_hyperparameter = {'parent': model_option,
                                     'value': model_item}
            cs.add_configuration_space(model_item, sub_configuration_space, parent_hyperparameter=parent_hyperparameter)
        return cs

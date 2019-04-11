from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter


class ComponentsManager(object):
    def get_hyperparameter_search_space(self, task_type, include=None, exclude=None):
        if task_type in ['binary', 'multiclass']:
            from alphaml.engine.components.models.classification import _classifiers
            builtin_models = _classifiers.keys()
            builtin_estimators = _classifiers
        elif task_type in ['img_binary', 'img_multiclass']:
            from alphaml.engine.components.models.image_classification import _img_classifiers
            builtin_models = _img_classifiers.keys()
            builtin_estimators = _img_classifiers
        else:
            raise ValueError('Undefined Task Type: %s' % task_type)

        model_candidates = set()
        if include is not None:
            for model in include:
                if model in builtin_models:
                    model_candidates.add(model)
                else:
                    raise ValueError("The estimator %s is NOT available in alpha-ml!" % str(model))
        else:
            model_candidates = set(builtin_models)

        if exclude is not None:
            for model in exclude:
                if model in model_candidates:
                    model_candidates.remove(model)

        return self.get_configuration_space(builtin_estimators, list(model_candidates))

    def get_configuration_space(self, builtin_estimators, model_candidates):
        """
        Reference: pipeline/base=325, classification/__init__=121
        """
        cs = ConfigurationSpace()
        # TODO: set the default model.
        model_option = CategoricalHyperparameter("estimator", model_candidates, default_value=model_candidates[0])
        cs.add_hyperparameter(model_option)

        for model_item in model_candidates:
            sub_configuration_space = builtin_estimators[model_item].get_hyperparameter_search_space()
            parent_hyperparameter = {'parent': model_option,
                                     'value': model_item}
            cs.add_configuration_space(model_item, sub_configuration_space, parent_hyperparameter=parent_hyperparameter)
        return cs

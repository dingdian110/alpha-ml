from alphaml.engine.components.componets_manager import ComponentsManager
from alphaml.engine.components.models.classification import _classifiers

if __name__ == "__main__":
    # print(_classifiers)
    # for item in _classifiers:
    #     name, cls = item, _classifiers[item]
    #     print(cls.get_hyperparameter_search_space())
    cs = ComponentsManager().get_hyperparameter_search_space(3)
    print(cs.get_default_configuration())

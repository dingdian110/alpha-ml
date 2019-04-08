
def test_configspace():
    from alphaml.engine.components.componets_manager import ComponentsManager
    from alphaml.engine.components.models.classification import _classifiers

    # print(_classifiers)
    # for item in _classifiers:
    #     name, cls = item, _classifiers[item]
    #     print(cls.get_hyperparameter_search_space())
    cs = ComponentsManager().get_hyperparameter_search_space(3)
    print(cs.sample_configuration(5))
    # print(cs.get_default_configuration())


def test_auto():
    from sklearn.datasets import load_breast_cancer
    from alphaml.engine.components.data_manager import DataManager
    from alphaml.estimators.classifier import Classifier

    X, y = load_breast_cancer(return_X_y=True)
    # Classifier(exclude_models=['libsvm_svc']).fit(DataManager(X, y))

    for _ in range(5):
        Classifier(include_models=['adaboost', 'gradient_boosting', 'random_forest'], optimizer='ts_smac').fit(DataManager(X, y))

    for _ in range(5):
        Classifier(include_models=['adaboost', 'gradient_boosting', 'random_forest'], optimizer='smac').fit(DataManager(X, y))


if __name__ == "__main__":
    test_auto()

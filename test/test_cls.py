
def test_configspace():
    from alphaml.engine.components.components_manager import ComponentsManager
    from alphaml.engine.components.models.classification import _classifiers

    # print(_classifiers)
    # for item in _classifiers:
    #     name, cls = item, _classifiers[item]
    #     print(cls.get_hyperparameter_search_space())
    cs = ComponentsManager().get_hyperparameter_search_space(3)
    print(cs.sample_configuration(5))
    # print(cs.get_default_configuration())


def load_data(name='iris'):
    from sklearn.datasets import load_breast_cancer
    from sklearn.datasets import load_iris
    if name == 'iris':
        X, y = load_iris(return_X_y=True)
    elif name == 'breast_cancer':
        X, y = load_breast_cancer(return_X_y=True)
    else:
        raise ValueError('Unsupported Dataset: %s' % name)
    return X, y


def test_auto():
    from alphaml.engine.components.data_manager import DataManager
    from alphaml.estimators.classifier import Classifier

    X, y = load_data('iris')
    # Classifier(exclude_models=['libsvm_svc']).fit(DataManager(X, y))

    # for _ in range(5):
    #     Classifier(include_models=['adaboost', 'gradient_boosting', 'random_forest'], optimizer='ts_smac').fit(DataManager(X, y))

    for _ in range(5):
        Classifier(include_models=['adaboost', 'gradient_boosting', 'random_forest'], optimizer='smac').fit(DataManager(X, y))


if __name__ == "__main__":
    test_auto()

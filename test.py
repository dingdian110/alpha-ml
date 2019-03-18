
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
    import sklearn
    from sklearn.datasets import load_breast_cancer
    from alphaml.estimators.classifier import Classifier

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
    Classifier().fit(X_train, y_train)


if __name__ == "__main__":
    test_auto()

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


def test_cash_module():
    from alphaml.engine.components.data_manager import DataManager
    from alphaml.estimators.classifier import Classifier
    from alphaml.datasets.cls_dataset.dataset_loader import load_data

    X, y, _ = load_data('dermatology')
    print(X.shape, y.shape)
    # Classifier(exclude_models=['libsvm_svc']).fit(DataManager(X, y))
    # Classifier(include_models=['adaboost', 'gradient_boosting', 'random_forest'],
    # optimizer='ts_smac').fit(DataManager(X, y))

    cls = Classifier(
        include_models=['adaboost', 'gradient_boosting', 'random_forest', 'gaussian_nb', 'k_nearest_neighbors'],
        optimizer='ts_smbo'
        ).fit(DataManager(X, y), metric='accuracy', runcount=100)
    print(cls.predict(X))


if __name__ == "__main__":
    test_cash_module()

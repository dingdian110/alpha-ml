import pandas as pd


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
    from sklearn.datasets import load_digits, load_wine, load_iris, load_breast_cancer
    if name == 'iris':
        # smbo: 0.
        # ts-smbo: 0.
        X, y = load_iris(return_X_y=True)
    elif name == 'breast_cancer':
        # smbo: 0.00877192982456143
        # ts-smbo: 0.01754385964912286
        X, y = load_breast_cancer(return_X_y=True)
    elif name == 'wine':
        # smbo: 0.
        # ts-smbo: 0.
        X, y = load_wine(return_X_y=True)
    elif name == 'digits':
        # smbo: 0.019444444444444486
        # ts-smbo: 0.01388888888888884
        X, y = load_digits(return_X_y=True)
    elif name == 'fall_detection':
        file_path = 'data/cls_data/fall_detection/falldeteciton.csv'
        data = pd.read_csv(file_path, delimiter=',').values
        return data[:, 1:], data[:, 0]
    else:
        raise ValueError('Unsupported Dataset: %s' % name)
    return X, y


def test_cash_module():
    from alphaml.engine.components.data_manager import DataManager
    from alphaml.estimators.classifier import Classifier
    from alphaml.datasets.cls_dataset.dataset_loader import load_data

    X, y, _ = load_data('dermatology')
    print(X.shape, y.shape)
    # Classifier(exclude_models=['libsvm_svc']).fit(DataManager(X, y))
    # Classifier(include_models=['adaboost', 'gradient_boosting', 'random_forest'],
    # optimizer='ts_smac').fit(DataManager(X, y))
    cls = Classifier(include_models=['adaboost', 'gradient_boosting', 'random_forest'],
                     optimizer='smbo'
                     ).fit(DataManager(X, y), metric='accuracy', runcount=10)
    print(cls.predict(X))


if __name__ == "__main__":
    test_cash_module()

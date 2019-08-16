import pandas as pd


# def load_data(name='iris'):
#     from sklearn.datasets import load_digits, load_wine, load_iris, load_breast_cancer
#     if name == 'iris':
#         # smbo: 0.
#         # ts-smbo: 0.
#         X, y = load_iris(return_X_y=True)
#     elif name == 'breast_cancer':
#         # smbo: 0.00877192982456143
#         # ts-smbo: 0.01754385964912286
#         X, y = load_breast_cancer(return_X_y=True)
#     elif name == 'wine':
#         # smbo: 0.
#         # ts-smbo: 0.
#         X, y = load_wine(return_X_y=True)
#     elif name == 'digits':
#         # smbo: 0.019444444444444486
#         # ts-smbo: 0.01388888888888884
#         X, y = load_digits(return_X_y=True)
#     elif name == 'fall_detection':
#         file_path = 'data/cls_data/fall_detection/falldeteciton.csv'
#         data = pd.read_csv(file_path, delimiter=',').values
#         return data[:, 1:], data[:, 0]
#     else:
#         raise ValueError('Unsupported Dataset: %s' % name)
#     return X, y

def test_cash_module():
    from alphaml.engine.components.data_manager import DataManager
    from alphaml.estimators.regressor import Regressor
    from alphaml.datasets.rgs_dataset.dataset_loader import load_data
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    df = pd.read_csv("data/rgs_data/boston/boston.csv", na_values=['?'])
    # Classifier(exclude_models=['libsvm_svc']).fit(DataManager(X, y))
    # Classifier(include_models=['adaboost', 'gradient_boosting', 'random_forest'],
    # optimizer='ts_smac').fit(DataManager(X, y))
    rgs = Regressor(include_models=['decision_tree', 'random_forest'],
                    optimizer='mono_smbo',
                    ).fit(df, metric='mae', runcount=10)
    # pred = rgs.predict(x_test)
    # print(mean_absolute_error(y_test, pred))
    # pred1, pred2 = cls.predict(x_test)
    # print("Prediction: Ensemble",pred1)
    # print("Prediction: Best",pred2)
    # print("Truth",y_test)
    # print("Test: Ensemble", mean_absolute_error(y_test, pred1))
    # print("Test: Best", mean_absolute_error(y_test, pred2))


if __name__ == "__main__":
    test_cash_module()

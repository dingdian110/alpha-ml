import pandas as pd

from alphaml.engine.components.data_preprocessing.encoder import one_hot

from sklearn.linear_model import LogisticRegression

import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


def test_cash_module():
    from alphaml.engine.components.data_manager import DataManager
    from alphaml.engine.components.feature_engineering.auto_cross import AutoCross
    from alphaml.estimators.classifier import Classifier
    from alphaml.datasets.cls_dataset.dataset_loader import load_data
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    X, y, _ = load_data('satimage')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    dm = DataManager(X_train, y_train)
    # print(af.transform(dm).train_X)
    # Classifier(exclude_models=['libsvm_svc']).fit(DataManager(X, y))
    # Classifier(include_models=['adaboost', 'gradient_boosting', 'random_forest'],
    # optimizer='ts_smac').fit(DataManager(X, y))
    cls = Classifier(include_models=['gradient_boosting'],
                     optimizer='mono_smbo',
                     # ensemble_method='ensemble_selection',
                     # ensemble_size=6,
                     ).fit(dm, metric='accuracy', runcount=5)
    pred = cls.predict(X_test)
    print("Test: Best", accuracy_score(y_test, pred))


if __name__ == "__main__":
    test_cash_module()

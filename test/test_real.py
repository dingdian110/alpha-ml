import sys

sys.path.append('/home/daim_gpu/sy/AlphaML')
from alphaml.engine.components.data_manager import DataManager
from alphaml.estimators.classifier import Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
from alphaml.engine.components.pipeline.data_preprocessing_operator import ImputerOperator, ScalerOperator


def test_cash_module():
    data = pd.read_csv("data/cls_data/santander/train.csv")
    dm = ImputerOperator().operate([data.drop(columns=["ID"])])
    dm.split(stratify=True)
    cls = Classifier(include_models=['xgboost', 'random_forest', 'libsvm_svc', 'decision_tree'],
                     optimizer='mono_smbo',
                     # ensemble_method='stacking',
                     # ensemble_size=6,
                     ).fit(dm, metric='auc', runcount=200)
    data = pd.read_csv("data/cls_data/santander/test.csv").values
    x_data = data[:, 1:]
    pred2 = cls.predict(x_data)

    import csv
    with open('data/cls_data/santander/submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'TARGET'])
        for i in range(len(pred2)):
            line = [int(data[i, 0]), pred2[i]]
            writer.writerow(line)


if __name__ == "__main__":
    test_cash_module()

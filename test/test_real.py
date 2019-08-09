import sys

sys.path.append('/home/daim_gpu/sy/AlphaML')
from alphaml.engine.components.data_manager import DataManager
from alphaml.estimators.classifier import Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
from alphaml.engine.components.pipeline.data_preprocessing_operator import ImputerOperator, ScalerOperator


def test_cash_module():
    df = pd.read_csv("data/cls_data/santander/train.csv")
    df = df.drop(columns=["ID"])
    cls = Classifier(include_models=['xgboost', 'random_forest', 'decision_tree'],
                     optimizer='mono_smbo',
                     # ensemble_method='ensemble_selection',
                     # ensemble_size=5,
                     ).fit(df, metric='auc', runcount=200)
    df = pd.read_csv("data/cls_data/santander/test.csv")
    data = df.values
    df = df.drop(columns=["ID"])
    pred2 = cls.predict(df)
    print(pred2)

    import csv
    with open('data/cls_data/santander/submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'TARGET'])
        for i in range(len(pred2)):
            line = [int(data[i, 0]), pred2[i]]
            writer.writerow(line)


if __name__ == "__main__":
    test_cash_module()

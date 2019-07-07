import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import warnings

from alphaml.datasets.cls_dataset.dataset_loader import load_data
from alphaml.engine.components.feature_engineering.auto_feature import AutoFeature
from alphaml.estimators.classifier import Classifier
from alphaml.engine.components.data_manager import DataManager

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--generated_feature", type=int, default=1)
args = parser.parse_args()

x, y, c = load_data("poker")

dm = DataManager(x, y)

lr = LogisticRegression()
lr.fit(dm.train_X, dm.train_y)
y_pred = lr.predict(dm.val_X)
print("original lr accu:", accuracy_score(dm.val_y, y_pred), flush=True)

if args.generated_feature == 1:
    af = AutoFeature(10)
    af.fit(dm)
    generated_train_data, generated_valid_data = af.transform(dm)

clf = Classifier()
clf.fit(dm, metric="accuracy", runcount=10)
print("generated data, alphaml:", clf.score(dm.val_X, dm.val_y))

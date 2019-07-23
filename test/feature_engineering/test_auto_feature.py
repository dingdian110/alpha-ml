import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import warnings

from alphaml.datasets.cls_dataset.dataset_loader import load_data
from alphaml.engine.components.feature_engineering.auto_feature import AutoFeature
from alphaml.estimators.classifier import Classifier
from alphaml.engine.components.data_manager import DataManager

from time import time

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--generated_feature", type=int, default=1)
parser.add_argument("--dataset", type=str)
args = parser.parse_args()

x, y, c = load_data(args.dataset)

dm = DataManager(x, y)

lr = LogisticRegression()
lr.fit(dm.train_X, dm.train_y)
y_pred = lr.predict(dm.val_X)
print("original lr accu:", accuracy_score(dm.val_y, y_pred), flush=True)

if args.generated_feature > 0:
    af = AutoFeature("accuracy", "auto_cross")
    af.fit(dm, args.generated_feature)
    dm = af.transform(dm)

clf = Classifier()
start_time = time()
clf.fit(dm, metric="accuracy", runcount=50)
print("alphaml time:", time() - start_time)
print("dataset:", args.dataset)
print("generated data:", args.generated_feature, ", alphaml score:", clf.score(dm.val_X, dm.val_y))

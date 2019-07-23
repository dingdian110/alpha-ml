import os
import pandas as pd
import warnings

from alphaml.engine.components.data_manager import DataManager
from alphaml.engine.components.feature_engineering.auto_feature import AutoFeature

warnings.filterwarnings("ignore")

home_path = os.path.expanduser('~')
train_path = os.path.join(home_path, "datasets/santander/train.csv")
test_path = os.path.join(home_path, "datasets/santander/test.csv")

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

df_train.drop(labels=["ID_code"], axis=1, inplace=True)
df_test.drop(labels=["ID_code"], axis=1, inplace=True)

x_train = df_train.drop(labels=["target"], axis=1).values
y_train = df_train["target"].values
x_test = df_test.values

del df_train
del df_test

dm = DataManager(x_train, y_train)
dm.test_X = x_test

auto_feature = AutoFeature(metrics="auc")
dm = auto_feature.fit(dm, generated_num=100)

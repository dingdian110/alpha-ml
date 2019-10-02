import os
import pandas as pd

from alphaml.engine.components.data_manager import DataManager
from alphaml.engine.components.feature_engineering_operator.encoder import one_hot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load the titanic dataset
home_path = os.path.expanduser('~')
train_path = os.path.join(home_path, "datasets/titanic/train.csv")
test_path = os.path.join(home_path, "datasets/titanic/test.csv")

dm = DataManager()
dm.load_train_csv(file_location=train_path, label_col="Survived")
dm.load_test_csv(file_location=test_path)

# auto detect the feature type of features and split the dataset

dm = one_hot(dm)

lr = LogisticRegression()
print(dm.train_X.shape)
lr.fit(dm.train_X, dm.train_y)
y_pred = lr.predict(dm.val_X)
print(accuracy_score(dm.val_y, y_pred))

y_pred = lr.predict(dm.test_X)
submission = pd.read_csv(os.path.join(home_path, "datasets/titanic/gender_submission.csv"))
submission["Survived"] = y_pred
submission.to_csv(os.path.join(home_path, "datasets/titanic/submission_6.csv"), index=False)

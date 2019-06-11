import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from alphaml.engine.components.data_preprocessing.imputer import impute_df

COL_TYPE = ["NUMERICAL", "DISCRETE", "CATEGORICAL", "TEXT"]


class DataManager(object):

    def __init__(self, train_X=None, train_y=None, val_X=None, val_y=None, val_size=0.2, random_state=42):
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y
        self.split_size = val_size
        self.random_seed = random_state
        self.col_type = []

        if train_X is not None and train_y is not None and (self.val_X is None or self.val_y is None):
            self.split(val_size, random_state)

        self.test_X = None
        self.test_y = None

    def split(self, val_size=0.2, random_state=42):
        assert self.train_X is not None and self.train_y is not None

        # Split input into train and val subsets.
        self.train_X,  self.val_X, self.train_y, self.val_y = train_test_split(
            self.train_X, self.train_y, test_size=val_size, random_state=random_state, stratify=self.train_y)

    def set_col_type(self):
        pass

    def load_train_csv(self, file_location, label_col=-1):
        data = impute_df(pd.read_csv(file_location)).values
        swap_data = data[:, -1]
        data[:, -1] = data[:, label_col]
        data[:, label_col] = swap_data
        self.train_X = data[:, :-1]
        self.train_y = data[:, -1]
        self.split(self.split_size, self.random_seed)

    def load_test_csv(self, file_location):
        self.test_X = pd.read_csv(file_location).values

    def load_train_libsvm(self, file_location):
        pass

    def load_test_libsvm(self, file_location):
        pass

    def set_testX(self, test_X):
        self.test_X = test_X

    def set_testy(self, test_y):
        self.test_y = test_y

    def get_val(self):
        return self.val_X, self.val_y

    def get_train(self):
        return self.train_X, self.train_y

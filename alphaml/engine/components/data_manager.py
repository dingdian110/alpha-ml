import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from alphaml.engine.components.data_preprocessing.imputer import impute_df

_COL_TYPE = ["Discrete", "Numerical", "Categorical", "One-Hot"]


class Transformer(object):
    def __init__(self, idx, transformer):
        self.col_index = idx
        self.transformer = transformer


class DataManager(object):
    """
    This class implements the wrapper for data used in the ML task.

    It finishes the following preprocesses:
    1) detect the type of each feature (numerical, categorical, textual, ...)
    2) process the raw features:
        e.g, one-hot the categorical features; impute the missing values;
             encode the textual feature.
       the `info` will record the transformer objects (e.g., label encoder, one-hot encoder)
       during these processing.
       when processing the test data, such objects will be used again.
    """

    # X,y should be None if using DataManager().load_csv(...)
    def __init__(self, X=None, y=None):
        self.info = dict()
        self.info['task_type'] = None
        self.info['preprocess_transforms'] = list()
        self.info['feature_type'] = list()
        self.train_X = np.array(X)
        self.train_y = np.array(y)

        if X is not None and y is not None:
            self.set_col_type(self.train_X, None)

        self.test_X = None
        self.test_y = None

    def set_col_type(self, data, label_col):
        col_num = data.shape[1]
        for col_id in range(col_num):
            if label_col is not None and (col_id == label_col or col_id - label_col == col_num):
                continue
            col = data[:, col_id]
            try:
                col_f = col.astype(np.float64)
                col_i = col_f.astype(np.int32)
                if all(col_f == col_i) is True:
                    self.info['feature_type'].append("Discrete")
                else:
                    self.info['feature_type'].append("Numerical")
            except:
                self.info['feature_type'].append("Categorical")

    def load_train_csv(self, file_location, label_col=-1, keep_default_na=True, na_values=None):
        df = impute_df(pd.read_csv(file_location, keep_default_na=keep_default_na, na_values=na_values))
        data = df.values
        # set the feature types
        if not self.info['feature_type']:
            self.set_col_type(data, label_col)
        swap_data = data[:, -1]
        data[:, -1] = data[:, label_col]
        data[:, label_col] = swap_data
        self.train_X = data[:, :-1]
        self.train_y = LabelEncoder().fit_transform(data[:, -1])

    def load_test_csv(self, file_location, keep_default_na=True, na_values=None):
        df = impute_df(pd.read_csv(file_location, keep_default_na=keep_default_na, na_values=na_values))
        self.test_X = df.values

    def set_testX(self, X_test):
        self.test_X = X_test

    def set_testy(self, y_test):
        self.test_y = y_test

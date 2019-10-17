import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from alphaml.engine.components.data_preprocessing.imputer import impute_df

_COL_TYPE = ["Float", "Discrete", "Categorical", "Text", "One-Hot"]


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
    def __init__(self, X, y):
        self.info = dict()
        self.info['task_type'] = None
        self.info['preprocess_transforms'] = list()
        self.X_train = X
        self.y_train = y

        self.feature_types = None

        self.X_test = None
        self.y_test = None

    def set_col_type(self, df, label_col):
        self.feature_types = []
        for col in list(df.columns):
            if label_col is not None and col == label_col:
                continue
            dtype = df[col].dtype
            if dtype in [np.int, np.int16, np.int32, np.int64]:
                self.feature_types.append("Discrete")
            elif dtype in [np.float, np.float16, np.float32, np.float64, np.float128, np.double]:
                self.feature_types.append("Float")
            elif dtype in [np.str, np.str_, np.string_, np.object]:
                self.feature_types.append("Categorical")
            else:
                raise TypeError("Unknown data type:", dtype)

    def load_train_csv(self, file_location, label_col=-1, keep_default_na=True, na_values=None):
        df = impute_df(pd.read_csv(file_location, keep_default_na=keep_default_na, na_values=na_values))
        # set the feature types
        if self.feature_types is None:
            self.set_col_type(df, df.columns[label_col])
        data = df.values

        swap_data = data[:, -1]
        data[:, -1] = data[:, label_col]
        data[:, label_col] = swap_data
        self.X_train = data[:, :-1]
        self.y_train = LabelEncoder().fit_transform(data[:, -1])

    def load_test_csv(self, file_location, keep_default_na=True, na_values=None):
        df = impute_df(pd.read_csv(file_location, keep_default_na=keep_default_na, na_values=na_values))
        # set the feature types
        if self.feature_types is None:
            self.set_col_type(df, None)
        self.test_X = df.values

    def load_train_libsvm(self, file_location):
        pass

    def load_test_libsvm(self, file_location):
        pass

    def set_testX(self, X_test):
        self.X_test = X_test

    def set_testy(self, y_test):
        self.y_test = y_test

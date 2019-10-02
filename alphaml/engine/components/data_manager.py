import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from alphaml.engine.components.feature_engineering_operator.imputer import impute_df

_COL_TYPE = ["Float", "Discrete", "Categorical", "Text", "One-Hot"]


class DataManager(object):

    def __init__(self, train_X=None, train_y=None, val_X=None, val_y=None, val_size=0.2, stratify=True, spilt=True,
                 random_state=42):
        self.train_X = np.array(train_X)
        self.train_y = np.array(train_y)
        self.val_X = val_X
        self.val_y = val_y
        self.split_size = val_size
        self.random_seed = random_state
        self.stratify = stratify
        self.feature_types = None

        if spilt and train_X is not None and train_y is not None and (self.val_X is None or self.val_y is None):
            self.split(val_size, random_state)

        self.test_X = None
        self.test_y = None

    def split(self, val_size=0.2, random_state=42, stratify=True):
        assert self.train_X is not None and self.train_y is not None

        # Split input into train and val subsets.
        if self.stratify and stratify:
            self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(
                self.train_X, self.train_y, test_size=val_size, random_state=random_state, stratify=self.train_y)
        else:
            self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(
                self.train_X, self.train_y, test_size=val_size, random_state=random_state)

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
        self.train_X = data[:, :-1]
        self.train_y = LabelEncoder().fit_transform(data[:, -1])
        self.split(self.split_size, self.random_seed)

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

    def set_testX(self, test_X):
        self.test_X = test_X

    def set_testy(self, test_y):
        self.test_y = test_y

    def get_val(self):
        return self.val_X, self.val_y

    def get_train(self):
        return self.train_X, self.train_y

    def get_test(self):
        return self.test_X, self.test_y

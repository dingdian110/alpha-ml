import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer


def _split_data(x, train_size, valid_size, test_size):
    return x[:train_size, :], \
           x[train_size: train_size + valid_size, :], \
           x[train_size + valid_size: train_size + valid_size + test_size, :]


def impute_categorical(col):
    return col.fillna(col.mode().iloc[0])


def impute_float(col) -> pd.Series:
    mean_val = col.mean()
    return col.fillna(mean_val)


def impute_discrete(col) -> pd.Series:
    mean_val = int(col.mean())
    return col.fillna(mean_val)


def impute_col(col, datatype) -> pd.Series:
    if datatype == "categorical":
        return impute_categorical(col)
    elif datatype == "float":
        return impute_float(col)
    elif datatype == "discrete":
        return impute_discrete(col)
    else:
        raise TypeError("Require datatype to be categorical, float or discrete")


def impute_df(df) -> pd.DataFrame:
    for col in list(df.columns):
        dtype = df[col].dtype
        if dtype in [np.int, np.int16, np.int32, np.int64]:
            df[col] = impute_col(df[col], "discrete")
        elif dtype in [np.float, np.float16, np.float32, np.float64, np.float128, np.double]:
            df[col] = impute_col(df[col], "float")
        elif dtype in [np.str, np.str_, np.string_, np.object]:
            df[col] = impute_col(df[col], "categorical")
        else:
            raise TypeError("Unknow data type:", dtype)
    return df


def impute_dm(dm, missing_str):
    feature_types = dm.feature_types
    continuous_index = [i for i in range(len(feature_types)) if feature_types[i] == "Float"]
    categorical_index = [i for i in range(len(feature_types)) if feature_types[i] == "Categorical"]
    discrete_index = [i for i in range(len(feature_types)) if feature_types[i] == "Discrete"]

    (train_x, _), (valid_x, _), (test_x, _) = dm.get_train(), dm.get_val(), dm.get_test()
    train_size = len(train_x)
    valid_size = 0
    test_size = 0

    if train_x is None:
        raise ValueError("train_x has no value!!!")
    if valid_x is not None and test_x is not None:
        x = np.concatenate([train_x, valid_x, test_x])
        valid_size = len(valid_x)
        test_size = len(test_x)
    elif valid_x is not None:
        x = np.concatenate([train_x, valid_x])
        valid_size = len(valid_x)
    else:
        x = train_x

    x = x.astype(np.object)
    # impute categorical values
    print("missing str:", missing_str)
    imputer = SimpleImputer(missing_values=missing_str, strategy="most_frequent")
    if len(categorical_index) > 0:
        x[:, categorical_index] = imputer.fit_transform(x[:, categorical_index])

    # impute discrete values
    imputer.strategy = "median"
    imputer.missing_values = np.nan
    if len(discrete_index) > 0:
        x[:, discrete_index] = imputer.fit_transform(x[:, discrete_index])

    # impute continuous values
    imputer.strategy = "mean"
    imputer.missing_values = np.nan
    if len(continuous_index) > 0:
        x[:, continuous_index] = imputer.fit_transform(x[:, continuous_index])

    train_x, valid_x, test_x = _split_data(x, train_size, valid_size, test_size)
    if valid_size == 0:
        valid_x = None
    if test_size == 0:
        test_x = None

    dm.train_X = train_x
    dm.val_X = valid_x
    dm.test_X = test_x

    return dm

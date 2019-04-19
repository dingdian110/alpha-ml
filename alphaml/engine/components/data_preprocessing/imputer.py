import pandas as pd
import numpy as np


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

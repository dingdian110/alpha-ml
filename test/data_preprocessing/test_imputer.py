import pandas as pd
import numpy as np

from alphaml.engine.components.data_preprocessing.imputer import impute_df, impute_dm
from alphaml.engine.components.data_manager import DataManager


def test_impute_df():
    df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f', 'h'],
                      columns=["one", "two", "three"])

    df["four"] = "bar"

    df2 = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    df2 = impute_df(df2)

    print("original df:")
    print(df)
    print("preprocessed df:")
    print(df2)


def test_impute_dm():
    train_x = np.array([["a", 1, "python", 4.5],
                        ["b", 2, "c++", 6.8],
                        ["c", 10, "java", 4.8]])

    valid_x = np.array([["a", 1, "scala", 4.5],
                        ["c", 2, "c++", 6.8],
                        ["d", 10, "python", 4.8]])

    test_x = np.array([["a", 1, "scala", 4.5]])

    train_x[2][0] = "???"
    train_x[2][2] = "???"
    valid_x[0][1] = np.nan
    test_x[0][-1] = np.nan

    dm = DataManager()

    dm.feature_types = ["Categorical", "Discrete", "Categorical", "Float"]

    dm.train_X = train_x.astype(np.object)
    dm.val_X = valid_x.astype(np.object)
    dm.test_X = test_x.astype(np.object)

    dm = impute_dm(dm, "???")

    print(dm.feature_types)
    print(dm.train_X)
    print("----------------------------")
    print(dm.val_X)
    print("----------------------------")
    print(dm.test_X)


if __name__ == '__main__':
    test_impute_dm()

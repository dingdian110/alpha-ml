import numpy as np

from alphaml.engine.components.data_manager import DataManager
from alphaml.engine.components.data_preprocessing.encoder import one_hot, bucketizer, categorical_indexer


def test_one_hot():
    train_x = np.array([["a", 1, "python", 4.5],
                        ["b", 2, "c++", 6.8],
                        ["c", 10, "java", 4.8]])

    valid_x = np.array([["a", 1, "scala", 4.5],
                        ["c", 2, "c++", 6.8],
                        ["d", 10, "python", 4.8]])

    test_x = np.array([["a", 1, "scala", 4.5]])

    dm = DataManager()

    dm.feature_types = ["Categorical", "Discrete", "Categorical", "Float"]

    dm.train_X = train_x
    dm.val_X = valid_x
    dm.test_X = test_x

    dm = one_hot(dm)

    print(dm.feature_types)
    print(dm.train_X)
    print("----------------------------")
    print(dm.val_X)
    print("----------------------------")
    print(dm.test_X)


def test_bucketizer():
    train_x = np.array([["a", 1, "python", 4.5],
                        ["b", 2, "c++", 6.8],
                        ["c", 10, "java", 4.8]])

    valid_x = np.array([["a", 1, "scala", 4.5],
                        ["c", 2, "c++", 6.8],
                        ["d", 10, "python", 4.8]])

    test_x = np.array([["a", 1, "scala", 4.5]])

    dm = DataManager()

    dm.feature_types = ["Categorical", "Discrete", "Categorical", "Float"]

    dm.train_X = train_x
    dm.val_X = valid_x
    dm.test_X = test_x

    dm = one_hot(dm)
    dm = bucketizer(dm)

    print(dm.feature_types)
    print(dm.train_X)
    print("----------------------------")
    print(dm.val_X)
    print("----------------------------")
    print(dm.test_X)


def test_categorical_indexer():
    train_x = np.array([["a", 1, "python", 4.5],
                        ["b", 2, "c++", 6.8],
                        ["c", 10, "java", 4.8]])

    valid_x = np.array([["a", 1, "scala", 4.5],
                        ["c", 2, "c++", 6.8],
                        ["d", 10, "python", 4.8]])

    test_x = np.array([["a", 1, "scala", 4.5]])

    dm = DataManager()

    dm.feature_types = ["Categorical", "Discrete", "Categorical", "Float"]

    dm.train_X = train_x
    dm.val_X = valid_x
    dm.test_X = test_x

    dm = categorical_indexer(dm)

    print(dm.feature_types)
    print(dm.train_X)
    print("----------------------------")
    print(dm.val_X)
    print("----------------------------")
    print(dm.test_X)


if __name__ == '__main__':
    # test_one_hot()
    # test_bucketizer()
    test_categorical_indexer()

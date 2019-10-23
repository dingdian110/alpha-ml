import numpy as np

from alphaml.engine.components.data_manager import DataManager
from alphaml.engine.components.data_preprocessing.scaler import standard_scale, minmax_scale, maxabs_scale, normalize


def test_standard(dm):
    dm = standard_scale(dm)

    print("after standard rescale\n")

    print(dm.train_X)
    print(dm.val_X)
    print(dm.test_X)
    print(dm.feature_types)


def test_minmax(dm):
    dm = minmax_scale(dm)

    print("after minmax rescale\n")

    print(dm.train_X)
    print(dm.val_X)
    print(dm.test_X)
    print(dm.feature_types)


def test_maxabs(dm):
    dm = maxabs_scale(dm)

    print("after minmax rescale\n")

    print(dm.train_X)
    print(dm.val_X)
    print(dm.test_X)
    print(dm.feature_types)


def test_normalize(dm):
    dm = normalize(dm)

    print("after normalize rescale\n")

    print(dm.train_X)
    print(dm.val_X)
    print(dm.test_X)
    print(dm.feature_types)


if __name__ == '__main__':
    np.random.seed(19941125)

    dm = DataManager()
    dm.train_X = np.random.rand(5, 5)
    dm.val_X = np.random.rand(3, 5)
    dm.test_X = np.random.rand(2, 5)
    dm.feature_types = ["Discrete", "One-Hot", "Float", "Float", "Categorical"]

    print("Original data......\n")
    print(dm.train_X)
    print(dm.val_X)
    print(dm.test_X)
    print(dm.feature_types)

    print("start test MinMaxScaler.......\n")
    test_minmax(dm)

    print("start test StandardScaler......\n")
    test_standard(dm)

    print("start test MaxAbsScaler......\n")
    test_maxabs(dm)

    print("start test L1 Normalize......\n")
    test_normalize(dm)

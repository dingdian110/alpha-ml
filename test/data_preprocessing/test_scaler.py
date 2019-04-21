import numpy as np

from alphaml.engine.components.data_manager import DataManager
from alphaml.engine.components.data_preprocessing.scaler import standard_rescale, minmax_rescale


def test_standard(dm):
    dm = standard_rescale(dm)

    print("after standard rescale\n")

    print(dm.train_X)
    print(dm.val_X)
    print(dm.test_X)
    print(dm.feature_types)


def test_minmax(dm):
    dm = minmax_rescale(dm)

    print("after minmax rescale\n")

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
    dm.feature_types = ["Discrete", "One-Hot", "Folat", "Float", "Categorical"]

    print("Original data......\n")
    print(dm.train_X)
    print(dm.val_X)
    print(dm.test_X)
    print(dm.feature_types)

    print("start test MinMaxScaler.......\n")
    test_minmax(dm)

    print("start test StandardScaler......\n")
    test_standard(dm)

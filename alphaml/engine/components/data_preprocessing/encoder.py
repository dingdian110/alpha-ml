import numpy as np

from alphaml.engine.components.data_manager import DataManager

from sklearn.preprocessing import OneHotEncoder


def _split_data(x, train_size, valid_size, test_size):
    return x[:train_size, :], \
           x[train_size: train_size + valid_size, :], \
           x[train_size + valid_size: train_size + valid_size + test_size, :]


def one_hot(dm: DataManager) -> DataManager:
    feature_types = dm.feature_types
    categorical_index = [i for i in range(len(feature_types)) if feature_types[i] == "Categorical"]
    other_index = [i for i in range(len(feature_types)) if feature_types[i] != "Categorical"]

    encoder = OneHotEncoder(handle_unknown="ignore")
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

    categorical_x = x[:, categorical_index]
    other_x = x[:, other_index]

    encoder.fit(categorical_x)
    categorical_x = encoder.transform(categorical_x).toarray()

    categorical_features = ["One-Hot"] * categorical_x.shape[1]
    other_features = [feature_types[i] for i in other_index]
    # print("categorical_x.shape =", categorical_x.__class__)
    # print("other_x.shape =", other_x.__class__)

    x = np.hstack((categorical_x, other_x)).astype(np.float)
    dm.feature_types = np.concatenate((categorical_features, other_features))

    train_x, valid_x, test_x = _split_data(x, train_size, valid_size, test_size)
    if valid_size == 0:
        valid_x = None
    if test_size == 0:
        test_x = None

    dm.train_X = train_x
    dm.val_X = valid_x
    dm.test_X = test_x

    return dm


def bucketizer(dm):
    return dm

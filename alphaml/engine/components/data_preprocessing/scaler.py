from sklearn.preprocessing import MinMaxScaler, StandardScaler


def minmax_rescale(dm):
    feature_types = dm.feature_types
    numercial_index = [i for i in range(len(feature_types))
                       if feature_types[i] == "Float" or feature_types[i] == "Discrete"]

    (train_x, _), (valid_x, _), (test_x, _) = dm.get_train(), dm.get_val(), dm.get_test()
    scaler = MinMaxScaler()

    train_x[:, numercial_index] = scaler.fit_transform(train_x[:, numercial_index])
    dm.train_X = train_x
    if valid_x is not None:
        valid_x[:, numercial_index] = scaler.transform(valid_x[:, numercial_index])
        dm.val_X = valid_x
    if test_x is not None:
        test_x[:, numercial_index] = scaler.transform(test_x[:, numercial_index])
        dm.test_X = test_x

    return dm


def standard_rescale(dm):
    feature_types = dm.feature_types
    numercial_index = [i for i in range(len(feature_types))
                       if feature_types[i] == "Float" or feature_types[i] == "Discrete"]

    (train_x, _), (valid_x, _), (test_x, _) = dm.get_train(), dm.get_val(), dm.get_test()
    scaler = StandardScaler()

    train_x[:, numercial_index] = scaler.fit_transform(train_x[:, numercial_index])
    dm.train_X = train_x
    if valid_x is not None:
        valid_x[:, numercial_index] = scaler.transform(valid_x[:, numercial_index])
        dm.val_X = valid_x
    if test_x is not None:
        test_x[:, numercial_index] = scaler.transform(test_x[:, numercial_index])
        dm.test_X = test_x

    return dm

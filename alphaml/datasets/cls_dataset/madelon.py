import pandas as pd
from sklearn import preprocessing


def load_madelon(data_folder):
    file_path = data_folder + 'phpfLuQE4.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    labels = data[:, -1] - 1
    features = data[:, :-1]
    min_max_scaler = preprocessing.MinMaxScaler()
    features = min_max_scaler.fit_transform(features)
    return features, labels

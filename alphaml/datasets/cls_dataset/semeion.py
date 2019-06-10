import pandas as pd


def load_semeion(data_folder):
    file_path = data_folder + 'phpfLuQE4.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    labels = data[:, -1] - 1
    features = data[:, :-1]
    return features, labels

import pandas as pd


def load_musk(data_folder):
    file_path = data_folder + 'musk.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, 3:-1]/625., data[:, -1]

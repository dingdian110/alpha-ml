import pandas as pd


def load_pendigits(data_folder):
    file_path = data_folder + 'dataset_32_pendigits.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, :-1], data[:, -1]

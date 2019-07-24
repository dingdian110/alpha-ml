import pandas as pd


def load_eucalyptus(data_folder):
    file_path = data_folder + 'eucalyptus.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, :-1], data[:, -1]

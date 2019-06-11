import pandas as pd


def load_dna(data_folder):
    file_path = data_folder + 'dna.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, :-1], data[:, -1] - 1

import pandas as pd
from alphaml.datasets.utils import trans_label

def load_electricity(data_folder):
    file_path = data_folder + 'electricity.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, :-1], trans_label(data[:, -1])

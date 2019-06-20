import pandas as pd
from alphaml.datasets.utils import trans_label


def load_kropt(data_folder):
    file_path = data_folder + 'dataset_188_kropt.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    label = trans_label(data[:, -1])
    print(data[:, 0])
    for c in range(6):
        if c % 2 == 0:
            data[:, c] = [(ord(t) - ord('a') + 1) for t in data[:, c]]
    return data[:, :-1], label


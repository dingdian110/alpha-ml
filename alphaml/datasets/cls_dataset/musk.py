import numpy as np
import pandas as pd


def load_musk(data_folder):
    file_path = data_folder + 'musk.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    X = data[:, 3:-1]/625.
    y = np.array([int(item) for item in data[:, -1]])
    return X, y

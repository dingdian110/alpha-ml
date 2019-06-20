import numpy as np
import pandas as pd


def load_pc4(data_folder):
    file_path = data_folder + 'pc4.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, :-1], np.array([int(t == True) for t in data[:, -1]])

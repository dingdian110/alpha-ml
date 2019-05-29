import numpy as np
import pandas as pd


def load_fri_c1(data_folder):
    file_path = data_folder + 'fri_c1_1000_25.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, :-1], np.array([int(t == 'P') for t in data[:, -1]])


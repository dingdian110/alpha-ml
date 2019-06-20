import numpy as np
import pandas as pd


def load_quake(data_folder):
    file_path = data_folder + 'quake.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, :-1], np.array([int(t == 'N') for t in data[:, -1]])


if __name__ == '__main__':
    x, y = load_quake('/home/thomas/PycharmProjects/alpha-ml/data/cls_data/quake/')
    print(x.shape)
    print(y.shape)
    print(set(y))
    print(x[:2])

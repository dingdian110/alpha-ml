import pandas as pd
import numpy as np


# def load_dermatology(data_folder):
#     data = pd.read_csv(data_folder+'dermatology.data', delimiter=',').values
#     return data[:, 1:], data[:, 0]
#     # L = []
#     # with open(data_folder+'dermatology.data', 'r') as f:
#     #     for line in f.readlines():
#     #         items = line.split('\n')[0].split(',')
#     #         l = []
#     #         for i in items:
#     #             l.append(int(i))
#     #         L.append(l)
#     # data = np.array(L)
#     # return data[:, 1:], data[:, 0]


def load_dermatology(data_folder):
    L = []
    file_path = data_folder+'dermatology.data'
    with open(file_path, 'r') as f:
        for line in f.readlines():
            items = line.split('\n')[0].split(',')
            l = []
            for i in items:
                if i.isdigit():
                    l.append(int(i))
                else:
                    l.append(0)
            L.append(l)
    data = np.array(L)
    return data[:, 1:], data[:, 0]

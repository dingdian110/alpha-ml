import pandas as pd
import numpy as np


def load_ova_breast(data_folder):
    file_path = data_folder + 'OVA_Breast.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, :-1], np.array([int(t != 'Other') for t in data[:, -1]])


# if __name__ == '__main__':
#     x, y = load_ova_breast('/home/thomas/PycharmProjects/alpha-ml/data/cls_data/ova_breast/')
#     print(x.shape)
#     print(y.shape)
#     print(set(y))
#     print(x[:2])

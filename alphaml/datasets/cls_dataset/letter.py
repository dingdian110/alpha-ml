import numpy as np
import pandas as pd


def load_letter(data_folder):
    file_path = data_folder + 'dataset_6_letter.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, :-1], np.array([(ord(t) - ord('A')) for t in data[:, -1]])


if __name__ == '__main__':
    x, y = load_letter('/home/thomas/PycharmProjects/alpha-ml/data/cls_data/letter/')
    print(x.shape)
    print(y.shape)
    print(set(y))
    print(x[:2])

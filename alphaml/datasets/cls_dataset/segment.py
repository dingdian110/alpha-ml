import pandas as pd

from alphaml.datasets.utils import trans_label


def load_segment(data_folder):
    file_path = data_folder + 'phpyM5ND4.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, :-1], trans_label(data[:, -1])


if __name__ == '__main__':
    x, y = load_segment('/home/thomas/PycharmProjects/alpha-ml/data/cls_data/segment/')
    print(x.shape)
    print(y.shape)
    print(set(y))
    print(x[:2])

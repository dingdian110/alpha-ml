import pandas as pd
# import sys
# sys.path.append('/home/thomas/PycharmProjects/alpha-ml')
from alphaml.datasets.utils import trans_label


def load_satimage(data_folder):
    file_path = data_folder + 'dataset_186_satimage.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, :-1], trans_label(data[:, -1])


# if __name__ == '__main__':
#     x, y = load_satimage('/home/thomas/PycharmProjects/alpha-ml/data/cls_data/satimage/')
#     print(x.shape)
#     print(y.shape)
#     print(set(y))
#     print(x[:2])

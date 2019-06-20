import pandas as pd


def load_optdigits(data_folder):
    file_path = data_folder + 'dataset_28_optdigits.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, :-1]/16., data[:, -1]


if __name__ == '__main__':
    x, y = load_optdigits('/home/thomas/PycharmProjects/alpha-ml/data/cls_data/optdigits/')
    print(x.shape)
    print(y.shape)
    print(set(y))
    print(x[:2])

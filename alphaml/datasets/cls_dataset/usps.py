import pandas as pd


def load_usps(data_folder):
    file_path = data_folder + 'usps.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, 1:], data[:, 0] - 1


if __name__ == '__main__':
    x, y = load_usps('/home/thomas/PycharmProjects/alpha-ml/data/cls_data/usps/')
    print(x.shape)
    print(y.shape)
    print(set(y))
    print(x[:2])

import pandas as pd


def load_mushroom(data_folder):
    file_path = data_folder + 'dataset_24_mushroom.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, 1:], data[:, 0]


if __name__ == '__main__':
    x, y = load_mushroom('/home/thomas/PycharmProjects/alpha-ml/data/cls_data/mushroom/')
    print(x[:2])

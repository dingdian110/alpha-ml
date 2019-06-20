import pandas as pd


def load_mnist(data_folder):
    file_path = data_folder + 'mnist_784.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, :-1]/255., data[:, -1]

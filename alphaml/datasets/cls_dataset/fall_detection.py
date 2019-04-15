import pandas as pd


def load_fall_detection(data_folder):
    file_path = data_folder + 'falldeteciton.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, 1:], data[:, 0]

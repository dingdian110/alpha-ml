import pandas as pd


def load_fall_detection():
    file_path = 'data/xgb_dataset/fall_detection/falldeteciton.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, 1:], data[:, 0]

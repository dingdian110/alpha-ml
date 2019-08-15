import pandas as pd


def load_covtype():
    file_path = 'data/xgb_dataset/covtype/covtype.data'
    data = pd.read_csv(file_path, delimiter=',', header=None).values
    return data[:, :-1], data[:, -1] - 1

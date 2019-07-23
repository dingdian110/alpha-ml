import pandas as pd

def load_higgs():
    file_path = 'data/xgb_dataset/higgs/higgs.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, 1:], data[:,0]
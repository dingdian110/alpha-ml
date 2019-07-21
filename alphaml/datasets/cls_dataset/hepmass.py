import pandas as pd

def load_hepmass():
    file_path = 'data/xgb_dataset/hepmass/hepmass.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, 1:], data[:,0]

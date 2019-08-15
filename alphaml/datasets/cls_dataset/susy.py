import pandas as pd

def load_susy():
    file_path = 'data/xgb_dataset/susy/susy.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, 1:], data[:,0]
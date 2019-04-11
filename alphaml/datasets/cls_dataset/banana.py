import os
import pandas as pd

def load_banana():
    print(os.curdir)
    file_path = 'data/xgb_dataset/banana/banana.csv'
    data = pd.read_csv(file_path, delimiter=',',converters={2: lambda x:int(int(x) == 1)}).values
    return data[:, :2], data[:, 2]
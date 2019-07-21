import os
import pandas as pd

def load_biomechanical2C():
    file_path = 'data/xgb_dataset/biomechanical/2C.csv'
    data = pd.read_csv(file_path, delimiter=',',converters={'class': lambda x:int(x == 'Normal')}).values
    return data[:, :6], data[:, 6]

def convert(x):
    if x == 'Spondylolisthesis':
        return 0
    elif x == 'Normal':
        return 1
    elif x == 'Hernia':
        return 2

def load_biomechanical3C():
    file_path = 'data/xgb_dataset/biomechanical/3C.csv'
    data = pd.read_csv(file_path, delimiter=',',converters={'class': convert}).values
    return data[:, :6], data[:, 6]

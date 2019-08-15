import os
import gc
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def load_talkinigdata():
    print(os.curdir)
    dtypes ={
        'ip':'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8'
    }
    to_read = ['ip', 'app', 'device', 'os', 'channel', 'is_attributed']
    to_parse = ['click_time']
    categorical_features = ['app', 'device', 'os', 'channel']
    file_path = 'data/xgb_dataset/talkingdata/train.csv'
    df = pd.read_csv(file_path, usecols=to_read, dtype=dtypes, nrows=1000000)
    clicks_by_ip = df.groupby(['ip']).size().rename('click_by_ip', inplace=True)
    print(df.groupby(['ip']))
    df = df.join(clicks_by_ip, on='ip')
    del clicks_by_ip
    gc.collect()
    del df['ip']

    for c in categorical_features:
        ohe = pd.get_dummies(df[c]).astype(np.int8)
        ohe.rename(columns=lambda x: c+str(x), inplace=True)
        df = df.join(ohe)
        del df[c]

    y = df['is_attributed'].values
    del df['is_attributed']
    X = df.values

    return X, y

if __name__ == '__main__':
    X, y = load_talkinigdata()
    print(X)
    print(y)


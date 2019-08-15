import pandas as pd
import numpy as np

def load_phishing():
    file_name= 'data/xgb_dataset/phishing/phishing.txt'
    df = pd.read_table(file_name, sep=',', header=None, prefix='feat', converters={30:lambda x: int(int(x)==1)})
    y = df['feat30'].values
    del df['feat30']

    for c in df.columns:
        ohe = pd.get_dummies(df[c]).astype(np.int8)
        ohe.rename(columns=lambda x: c+'_'+str(x), inplace=True)
        df = df.join(ohe)
        del df[c]

    X = df.values

    return X, y
if __name__ == '__main__':
    X, y = load_phishing()
    print(X)
import gc
import numpy as np
import pandas as pd
import xgboost as xgb

from pandas.core.categorical import Categorical
from scipy.sparse import csr_matrix, hstack

dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8'
}
to_read = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
to_parse = ['click_time']

# Features used in training
categorical_features = ['app', 'device', 'os', 'channel']
numerical_features = ['clicks_by_ip']



def sparse_dummies(df, column):
    '''Returns sparse OHE matrix for the column of the dataframe'''
    print(column)
    categories = Categorical(df[column])
    print(categories)
    # return a CategoricalDtype object
    column_names = np.array(["{}_{}".format(column, str(i)) for i in range(len(categories.categories))])
    print(column_names)
    # f-string, format strings
    N = len(categories)
    row_numbers = np.arange(N, dtype = np.int)
    ones = np.ones((N,))
    # categories.codes encode the strinig with number
    # create a matrix with 1's only at (i, i's category)
    return csr_matrix((ones, (row_numbers, categories.codes))), column_names

df_train = pd.read_csv('../../../data/click_fraud/train.csv', nrows=10000000, usecols = to_read, dtype = dtypes, parse_dates = to_parse)

print(df_train)

train_size = int(0.8 * df_train.shape[0])

print(df_train.groupby(['ip']).size())

clicks_by_ip = df_train.groupby(['ip']).size().rename('clicks_by_ip', inplace=True)
# 按照ip值分组并统计个数

df_train = df_train.join(clicks_by_ip, on='ip')     # add a feature clicks_by_ip

print(df_train)

del clicks_by_ip
gc.collect()    # memory collection

matrices = []
all_column_names = []
# create a matrix per categorical feature
for c in categorical_features:
    matrix, column_names = sparse_dummies(df_train, c)
    matrices.append(matrix)
    all_column_names.append(column_names)

# append a matrix for numerical features (one column per feature)
matrices.append(csr_matrix(df_train[numerical_features].values, dtype = float))
all_column_names.append(df_train[numerical_features].columns.values)

train_sparse = hstack(matrices, format = "csr")
feature_names = np.concatenate(all_column_names)
del matrices, all_column_names

X = train_sparse
print(X)
y = df_train['is_attributed']

del df_train
gc.collect()



# Create binary training and validation files for XGBoost
x1, y1 = X[:train_size], y.iloc[:train_size]
dm1 = xgb.DMatrix(x1, y1, feature_names=feature_names)
dm1.save_binary('../../../data/click_fraud/train_sample.bin')
del dm1, x1, y1
gc.collect()

x2, y2 = X[train_size:], y.iloc[train_size:]
dm2 = xgb.DMatrix(x2, y2, feature_names=feature_names)
dm2.save_binary('../../../data/click_fraud/validate_sample.bin')
del dm2, x2, y2
del X, y, train_sparse
gc.collect()

f = open('../../../data/click_fraud/feature_names.txt', 'w')
for i in feature_names:
    f.write(str(i))
    f.write('\n')
f.close()


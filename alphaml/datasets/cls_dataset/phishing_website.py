import gc
import numpy as np
import pandas as pd
import xgboost as xgb

from pandas.core.categorical import Categorical
from scipy.sparse import csr_matrix, hstack

categorical_features = ['having_IP_Address','URL_Length','Shortining_Service','having_At_Symbol','double_slash_redirecting','Prefix_Suffix','having_Sub_Domain','SSLfinal_State','Domain_registeration_length','Favicon','port','HTTPS_token','Request_URL','URL_of_Anchor','Links_in_tags','SFH','Submitting_to_email','Abnormal_URL','Redirect','on_mouseover','RightClick','popUpWidnow','Iframe','age_of_domain','DNSRecord','web_traffic','Page_Rank','Google_Index','Links_pointing_to_page','Statistical_report']
numerical_features = []

column_names = ['having_IP_Address','URL_Length','Shortining_Service','having_At_Symbol','double_slash_redirecting','Prefix_Suffix','having_Sub_Domain','SSLfinal_State','Domain_registeration_length','Favicon','port','HTTPS_token','Request_URL','URL_of_Anchor','Links_in_tags','SFH','Submitting_to_email','Abnormal_URL','Redirect','on_mouseover','RightClick','popUpWidnow','Iframe','age_of_domain','DNSRecord','web_traffic','Page_Rank','Google_Index','Links_pointing_to_page','Statistical_report','Result']

def sparse_dummies(df, column):
    '''Returns sparse OHE matrix for the column of the dataframe'''
    categories = Categorical(df[column])
    column_names = np.array(["{}_{}".format(column, str(i)) for i in range(len(categories.categories))])
    N = len(categories)
    row_numbers = np.arange(N, dtype=np.int)
    ones = np.ones((N,))
    return csr_matrix((ones, (row_numbers, categories.codes))), column_names

data = np.loadtxt('../../../data/phishing_website/train.txt', dtype=int, delimiter=',', converters={30: lambda x: int(int(x) == 1)})
df_train = pd.DataFrame(data, columns=column_names)
print(df_train)

train_size = int(0.8 * df_train.shape[0])

matrices = []
all_column_names = []
# create a matrix per categorical feature
for c in categorical_features:
    matrix, column_names = sparse_dummies(df_train, c)
    matrices.append(matrix)
    all_column_names.append(column_names)
    # print(column_names)

# append a matrix for numerical features (one column per feature)
matrices.append(csr_matrix(df_train[numerical_features].values, dtype=float))
all_column_names.append(df_train[numerical_features].columns.values)

train_sparse = hstack(matrices, format="csr")
feature_names = np.concatenate(all_column_names)
del matrices, all_column_names

X = train_sparse
y = df_train['Result']

del df_train
gc.collect()

# Create binary training and validation files for XGBoost
x1, y1 = X[:train_size], y.iloc[:train_size]
dm1 = xgb.DMatrix(x1, y1, feature_names=feature_names)
dm1.save_binary('../../../data/phishing_website/train_sample.bin')
del dm1, x1, y1
gc.collect()

x2, y2 = X[train_size:], y.iloc[train_size:]
dm2 = xgb.DMatrix(x2, y2, feature_names=feature_names)
dm2.save_binary('../../../data/phishing_website/validate_sample.bin')
del dm2, x2, y2
del X, y, train_sparse
gc.collect()

f = open('../../../data/phishing_website/feature_names.txt', 'w')
for i in feature_names:
    f.write(str(i))
    f.write('\n')
f.close()


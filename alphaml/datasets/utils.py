from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


def trans_label(input):
    le = preprocessing.LabelEncoder()
    le.fit(input)
    return le.transform(input)


def one_hot(input):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(input)
    return enc.transform(input)

import pandas as pd
import numpy as np

from alphaml.datasets.utils import one_hot
from alphaml.datasets.utils import trans_label


def load_credit_g(data_folder):
    file_path = data_folder + 'credit-g.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    feature = data[:, :-1]
    feature_num = feature.shape[1]
    feature_indices = list(range(feature_num))
    numerical_indices = [1, 4, 7, 10, 12, 15, 17]
    one_hot_indices = [i for i in feature_indices if i not in numerical_indices]
    one_hot_feature = one_hot(data[:, one_hot_indices]).toarray()
    print(one_hot_feature.shape[1])
    feature = np.hstack((one_hot_feature, feature[:, numerical_indices]))
    return feature, trans_label(data[:, -1])

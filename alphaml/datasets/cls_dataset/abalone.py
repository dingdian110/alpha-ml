import pandas as pd
from alphaml.datasets.utils import trans_label
from alphaml.datasets.utils import one_hot
import numpy as np


def load_abalone(data_folder):
    file_path = data_folder + 'abalone.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    label = trans_label(data[:, -1])
    one_hot_data = one_hot(data[:, 0:1]).toarray()
    feature = np.hstack((one_hot_data, data[:, 1:-1]))
    return feature, label

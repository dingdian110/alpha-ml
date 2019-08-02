import pandas as pd
from alphaml.datasets.utils import trans_label
from alphaml.datasets.utils import one_hot
import numpy as np


def load_amazon(data_folder):
    file_path = data_folder + 'amazon.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, :-1], trans_label(data[:, -1])

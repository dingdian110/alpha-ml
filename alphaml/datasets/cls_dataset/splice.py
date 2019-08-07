import numpy as np
import pandas as pd
from alphaml.datasets.utils import trans_label, one_hot


def load_splice(data_folder):
    file_path = data_folder + 'dataset_46_splice.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    label = trans_label(data[:, -1])
    transformed_data = [trans_label(item) for item in data[:, 1:-1].T]
    transformed_data = np.array(transformed_data).T
    transformed_data = one_hot(transformed_data).toarray()
    return transformed_data, label

import pandas as pd
from alphaml.datasets.utils import trans_label
from alphaml.engine.components.pipeline.data_preprocessing_operator import ImputerOperator


def load_higgs(data_folder):
    file_path = data_folder + 'higgs.csv'
    df = pd.read_csv(file_path, delimiter=',', na_values=['?'])
    op = ImputerOperator(label_col=0)
    dm = op.operate([df])
    return dm.train_X, trans_label(dm.train_y)

import pandas as pd


def load_waveform(data_folder):
    file_path = data_folder + 'dataset_60_waveform-5000.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    return data[:, :-1], data[:, -1]


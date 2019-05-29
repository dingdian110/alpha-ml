import pandas as pd


def load_shuttle(data_folder):
    file_path = data_folder + 'shuttle.csv'
    data = pd.read_csv(file_path, delimiter=',').values
    label =  data[:, -1] - 1
    import sklearn.preprocessing as pre
    scaler = pre.MinMaxScaler()
    data = scaler.fit_transform(data[:, :-1])
    return data, label


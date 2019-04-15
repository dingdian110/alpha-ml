import numpy as np


def load_australian(data_folder):
    L = []
    file_path = data_folder + 'australian_scale'
    with open(file_path, 'r') as f:
        for line in f.readlines():
            items = line.strip().split('\n')[0].split(' ')
            d = [0] * 15
            d[0] = int(int(items[0]) == 1)
            for item in items[1:]:
                key, value = item.split(':')
                d[int(key)] = float(value)
            L.append(d)
        data = np.array(L)
        return data[:, 1:], data[:, 0]

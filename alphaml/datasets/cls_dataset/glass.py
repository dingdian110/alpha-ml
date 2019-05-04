import numpy as np


def load_glass(data_folder):
    L = []
    file_path = data_folder + 'glass.scale'
    with open(file_path, 'r') as f:
        for line in f.readlines():
            items = line.strip().split('\n')[0].split(' ')
            d = [0] * 10
            label = int(items[0])
            d[0] = label - 1 if label < 4 else label - 2
            for item in items[1:]:
                key, value = item.split(':')
                d[int(key)] = float(value)
            L.append(d)
        data = np.array(L)
        return data[:, 1:], data[:, 0]

import numpy as np


def load_a9a(data_folder):
    L = []
    file_path = data_folder + 'phpwCsLLW.csv'
    with open(file_path, 'r') as f:
        first_line = True
        for line in f.readlines():
            if first_line:
                first_line = False
                continue
            line = line.strip()
            line = line[1:-1]
            items = line.split(' ')
            d = [0] * 125
            d[int(items[0])] = 1
            for item in items[1:-1]:
                value, key = item.split(',')
                d[int(key)] = float(value)
            d[-1] = int(int(items[-1]) == 1)
            L.append(d)
        data = np.array(L)
        return data[:, :-1], data[:, -1]

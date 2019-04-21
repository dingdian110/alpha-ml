import numpy as np


def load_usps(data_folder):
    L = []
    file_path = data_folder + 'usps'
    with open(file_path, 'r') as f:
        for line in f.readlines():
            items = line.split('\n')[0].split(' ')
            del items[257]
            l = [0] * 257
            l[0] = int(items[0]) - 1
            for item in items[1:]:
                if len(item.split(':')) < 2:
                    continue
                index, val = item.split(':')
                l[int(index)] = float(val)
            L.append(l)
    data = np.array(L)
    return data[:, 1:], data[:, 0]

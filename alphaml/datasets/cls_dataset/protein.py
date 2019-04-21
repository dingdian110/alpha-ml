import numpy as np


def load_protein(data_folder):
    L = []
    file_path = data_folder + 'protein'
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.replace(r' .', '0.')
            items = line.strip().split()
            d = [0] * 358
            d[0] = int(items[0])
            for item in items[1:]:
                if len(item.split(':')) < 2:
                    continue
                key, value = item.split(':')
                d[int(key)] = float(value)
            L.append(d)
        data = np.array(L)
        return data[:, 1:], data[:, 0]

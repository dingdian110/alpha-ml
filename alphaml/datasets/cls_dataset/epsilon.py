import numpy as np


def load_epsilon(data_folder, size=100000):
    L = []
    file_path = data_folder + 'epsilon_normalized'
    cnt = 0
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if not line or cnt >= size:
                break
            cnt += 1
            print(cnt)
            items = line.split('\n')[0].split(' ')
            l = list()
            l.append(int(int(items[0]) == 1))
            for item in items[1:]:
                value = item.split(':')[1]
                l.append(float(value))
            L.append(l)
    data = np.array(L)
    return data[:, 1:], data[:, 0]

import numpy as np


def load_gisette(data_folder):
    L = []
    file_path = data_folder + 'gisette.csv'
    with open(file_path, 'r') as f:
        first_line = True
        for line in f.readlines():
            if first_line:
                first_line = False
                continue
            line = line.strip()
            line = line[1:-1]
            items = line.split(' ')
            d = [0] * 5001
            for item in items[1:-1]:
                value, key = item.split(',')
                d[int(key)-1] = float(value)
            d[-1] = int(int(items[-1]) == 1)
            L.append(d)
        data = np.array(L)
        return data[:, :-1], data[:, -1]


if __name__ == '__main__':
    x, y = load_gisette('/home/thomas/PycharmProjects/alpha-ml/data/cls_data/gisette/')
    print(x.shape)
    print(y.shape)
    print(set(y))
    print(x[:2])

import numpy as np
from alphaml.datasets.utils import trans_label


def load_fbis_wc(data_folder):
    L = []
    file_path = data_folder + 'fbis.wc.csv'
    with open(file_path, 'r') as f:
        label = list()
        first_line = True
        for line in f.readlines():
            if first_line:
                first_line = False
                continue
            line = line.strip()
            line = line[1:-1]
            items = line.split(' ')
            d = [0] * 2000
            if items[0] != '':
                d[0] = int(items[0])
            if items[-1] == '':
                continue
            label.append(items[-1])
            for item in items[1:-1]:
                value, key = item.split(',')
                d[int(key)-1] = float(value)
            L.append(d)
        data = np.array(L)
        return data, trans_label(label)


# if __name__ == '__main__':
#     x, y = load_fbis_wc('/home/thomas/PycharmProjects/alpha-ml/data/cls_data/fbis_wc/')
#     print(x.shape)
#     print(y.shape)
#     print(set(y))
#     print(x[:2])

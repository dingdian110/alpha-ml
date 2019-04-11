import numpy as np

def load_usps():
    L = []
    file_path = 'data/xgb_dataset/usps/usps.txt'
    with open(file_path, 'r') as f:
        for line in f.readlines():
            items = line.split('\n')[0].split(' ')
            del items[257]
            l = []
            l.append(int(items[0])-1)
            del items[0]
            for i in items:
                if i == []:
                    continue
                value= i.split(':')[1]
                l.append(value)
            L.append(l)
    data = np.array(L)
    return data[:,1:], data[:,0]

if __name__ == '__main__':
    print(load_usps())

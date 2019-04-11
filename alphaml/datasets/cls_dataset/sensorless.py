import numpy as np

def load_sensorless():
    L = []
    file_path = 'data/xgb_dataset/sensorless/Sensorless.txt'
    with open(file_path, 'r') as f:
        for line in f.readlines():
            items = line.split('\n')[0].split(' ')
            l = []
            l.append(int(int(items[0])-1))
            del items[0]
            for i in items:
                if i == []:
                    continue
                value= i.split(':')[1]
                l.append(float(value))
            L.append(l)
    data = np.array(L)
    np.random.shuffle(data)
    return data[:,1:], data[:,0]

if __name__ == '__main__':
    X, y = load_sensorless()
    print(set(y))

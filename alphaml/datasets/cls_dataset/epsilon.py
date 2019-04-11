import numpy as np

def load_epsilon():
    L = []
    file_path = 'data/xgb_dataset/epsilon/epsilon_normalized.txt'
    cnt = 0
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if cnt >= 40000:
                break
            cnt = cnt + 1
            items = line.split('\n')[0].split(' ')
            l = []
            l.append(int(int(items[0])== 1))
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
    X, y = load_epsilon()
    print(X.shape)

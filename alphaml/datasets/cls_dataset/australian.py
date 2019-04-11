import pandas as pd

def load_australian():
    L = []
    file_path = 'data/xgb_dataset/australian/australian_scale.txt'
    with open(file_path, 'r') as f:
        for line in f.readlines():
            items = line.strip().split('\n')[0].split(' ')
            d ={}
            d['label'] = int(int(items[0])==1)
            del items[0]
            for item in items:
                key, value = item.split(':')
                d[key] = float(value)
            L.append(d)
        df = pd.DataFrame(L)
        y = df['label'].values
        del df['label']
        X = df.values
        return X, y
if __name__ == '__main__':
    X, y = load_australian()
    print(X)
    print(set(y))
import pandas as pd

def load_protein():
    L = []
    file_path = 'data/xgb_dataset/protein/protein'
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.replace(r' .', '0.')
            items = line.strip().split('\n')[0].strip().split()
            d ={}
            d['label'] = int(items[0])
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
    X, y = load_protein()
    print(X)
    print(y)
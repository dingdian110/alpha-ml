import pandas as pd


def load_a9a(data_folder):
    L = []
    file_path = data_folder + 'a9a'
    with open(file_path, 'r') as f:
        for line in f.readlines():
            items = line.strip().split('\n')[0].split(' ')
            d ={}
            d['label'] = int(int(items[0]) == 1)
            for item in items[1:]:
                if len(item.split(':')) < 2:
                    continue
                key, value = item.split(':')
                d[key] = int(value)
            L.append(d)
        df = pd.DataFrame(L)
        y = df['label'].values
        del df['label']
        df.fillna(0, inplace=True)
        X = df.values
        return X, y


if __name__ == '__main__':
    X, y = load_a9a('data/cls_data/a9a/')
    print(X.shape)
    print(y.shape)
    print(X[:5])
    print(y[:5])

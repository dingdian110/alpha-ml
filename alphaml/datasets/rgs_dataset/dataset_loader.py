data_dir_template = 'data/rgs_data/%s/'


def load_data(dataset_name):
    if dataset_name == 'boston':
        from sklearn.datasets import load_boston
        boston=load_boston()
        X, y = boston.data, boston.target
    else:
        raise ValueError('Invalid dataset name: %s!' % dataset_name)
    print(X.shape, y.shape)
    print(min(y), max(y))
    return X, y, None

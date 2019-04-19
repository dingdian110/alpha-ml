data_dir_template = 'data/cls_data/%s/'


def load_data(dataset_name):
    if dataset_name == 'iris':
        from sklearn.datasets import load_iris
        iris = load_iris()
        X, y = iris.data, iris.target
        num_cls = 3
    elif dataset_name == 'digits':
        from sklearn.datasets import load_digits
        digits = load_digits()
        X, y = digits.data, digits.target
        num_cls = 10
    elif dataset_name == 'wine':
        from sklearn.datasets import load_wine
        wine = load_wine()
        X, y = wine.data, wine.target
        num_cls = 3
    elif dataset_name == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X, y = data.data, data.target
        num_cls = 2
    elif dataset_name == 'fall_detection':
        from alphaml.datasets.cls_dataset.fall_detection import load_fall_detection
        X, y = load_fall_detection(data_dir_template % dataset_name)
        num_cls = 6
    elif dataset_name == 'dermatology':
        from alphaml.datasets.cls_dataset.dermatology import load_dermatology
        X, y = load_dermatology(data_dir_template % dataset_name)
        num_cls = 4
    elif dataset_name == 'a9a':
        from alphaml.datasets.cls_dataset.a9a import load_a9a
        X, y = load_a9a(data_dir_template % dataset_name)
        num_cls = 2
    elif dataset_name == 'usps':
        from alphaml.datasets.cls_dataset.usps import load_usps
        X, y = load_usps(data_dir_template % dataset_name)
        num_cls = 10
    elif dataset_name == 'protein':
        from alphaml.datasets.cls_dataset.protein import load_protein
        X, y = load_protein(data_dir_template % dataset_name)
        num_cls = 3
    elif dataset_name == 'australian_scale':
        from alphaml.datasets.cls_dataset.australian import load_australian
        X, y = load_australian(data_dir_template % dataset_name)
        num_cls = 2
    elif dataset_name == 'poker':
        from alphaml.datasets.cls_dataset.poker import load_poker
        X, y = load_poker(data_dir_template % dataset_name)
        num_cls = 10
    elif dataset_name == 'shuttle':
        from alphaml.datasets.cls_dataset.shuttle import load_shuttle
        X, y = load_shuttle(data_dir_template % dataset_name)
        num_cls = 7
    elif dataset_name == 'sensit_vehicle':
        from alphaml.datasets.cls_dataset.sensit_vehicle import load_sensit_vehicle
        X, y = load_sensit_vehicle(data_dir_template % dataset_name)
        num_cls = 7
    elif dataset_name == 'connect_4':
        from alphaml.datasets.cls_dataset.connect_4 import load_connect_4
        X, y = load_connect_4(data_dir_template % dataset_name)
        num_cls = 3
    elif dataset_name == 'epsilon':
        from alphaml.datasets.cls_dataset.epsilon import load_epsilon
        X, y = load_epsilon(data_dir_template % dataset_name)
        num_cls = 2
    else:
        raise ValueError('Invalid dataset name!')
    print(X.shape, y.shape)
    print(min(y), max(y), num_cls)
    return X, y, num_cls

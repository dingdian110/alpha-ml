import numpy as np


def load_data(dataset_name):
    if dataset_name == 'cifar10':
        from alphaml.datasets.img_cls_dataset.cifar10 import load_cifar10
        X, y = load_cifar10()
        num_cls = 10
    elif dataset_name == 'dogs_vs_cats':
        from alphaml.datasets.img_cls_dataset.dogs_vs_cats import load_dogs_cats
        num_cls = 1 # 2
        pass
    else:
        raise ValueError('Invalid dataset name!')
    return X, y, num_cls

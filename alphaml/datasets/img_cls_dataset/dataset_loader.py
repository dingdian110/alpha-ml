import numpy as np


def load_data(dataset_name):
    if dataset_name == 'cifar10':
        from alphaml.datasets.img_cls_dataset.cifar10 import load_cifar10
        train_x, train_y, test_x, test_y = load_cifar10()
        num_cls = 10
    elif dataset_name == 'cifar100':
        from alphaml.datasets.img_cls_dataset.cifar100 import load_cifar100
        train_x, train_y, test_x, test_y = load_cifar100()
        num_cls = 100
    else:
        raise ValueError('Invalid dataset name!')
    return train_x, train_y, test_x, test_y, num_cls


def load_data_img(dataset_name):
    if dataset_name == 'cifar10':
        from alphaml.datasets.img_cls_dataset.cifar10 import load_cifar10_img
        trainpath, testpath = load_cifar10_img()
    elif dataset_name == 'cifar100':
        from alphaml.datasets.img_cls_dataset.cifar100 import load_cifar100_img
        trainpath, testpath = load_cifar100_img()
    elif dataset_name == 'dogs_vs_cats':
        from alphaml.datasets.img_cls_dataset.dogs_vs_cats import load_dogs_vs_cats_img
        trainpath, testpath = load_dogs_vs_cats_img()
    else:
        raise ValueError('Invalid dataset name!')
    return trainpath, testpath

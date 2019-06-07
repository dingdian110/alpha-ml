import numpy as np
import pickle
import os


def load_cifar100():
    packagepath = os.path.join('data', 'img_cls_data', 'cifar100')
    # train data
    file = open(os.path.join(packagepath, 'train_batch'), 'rb')
    dict = pickle.load(file, encoding='bytes')
    train_x = dict[b'data']
    train_x = np.reshape(train_x, [50000, 3, 32, 32]).transpose([0, 2, 3, 1])
    train_y = dict[b'fine_labels']
    train_y = np.array(train_y)

    # test data
    file = open(os.path.join(packagepath, 'test_batch'), 'rb')
    dict = pickle.load(file, encoding='bytes')
    test_x = dict[b'data']
    test_x = np.reshape(test_x, [10000, 3, 32, 32]).transpose([0, 2, 3, 1])
    test_y = dict[b'fine_labels']
    test_y = np.array(test_y)

    return train_x, train_y, test_x, test_y


def load_cifar100_img():
    traindir = os.path.join('data', 'img_cls_data', 'cifar100', 'train')
    testdir = os.path.join('data', 'img_cls_data', 'cifar100', 'test')
    return traindir, testdir

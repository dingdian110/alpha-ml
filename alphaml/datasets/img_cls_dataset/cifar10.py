import numpy as np
import pickle
import os


def load_cifar10():
    packagepath = os.path.join('data', 'img_cls_data', 'cifar10')
    # train data
    train_x = []
    train_y = []
    for i in range(1, 6):
        file = open(os.path.join(packagepath, 'data_batch_' + str(i)), 'rb')
        dict = pickle.load(file, encoding='bytes')
        train_batch_x = dict[b'data']
        train_batch_x = np.reshape(train_batch_x, [10000, 3, 32, 32]).transpose([0, 2, 3, 1])
        train_x.extend(train_batch_x)
        train_batch_y = dict[b'labels']
        train_y.extend(train_batch_y)
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    return train_x, train_y

    # # test data
    # test_x = []
    # test_y = []
    # file = open(os.path.join(packagepath, 'test_batch'), 'rb')
    # dict = pickle.load(file, encoding='bytes')
    # test_x = dict[b'data']
    # test_x = np.reshape(test_x, [10000, 3, 32, 32]).transpose([0, 2, 3, 1])
    # test_y = dict[b'labels']
    # test_y = np.array(test_y)

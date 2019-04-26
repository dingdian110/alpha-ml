import os

def load_dogs_vs_cats_img():
    traindir = os.path.join('data', 'img_cls_data', 'dogs_vs_cats', 'train')
    testdir = os.path.join('data', 'img_cls_data', 'dogs_vs_cats', 'test')
    return traindir, testdir
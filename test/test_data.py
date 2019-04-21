from alphaml.datasets.img_cls_dataset import load_data
x, y, _ =load_data('cifar10')
print(x)
print(x.shape)

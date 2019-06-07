import numpy as np
from keras import backend
import pickle
import os
from scipy import misc
try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

def array_to_img(x, data_format=None, scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
            either "channels_first" or "channels_last".
        scale: Whether to rescale image values
            to be within `[0, 255]`.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=backend.floatx())
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 4:
        # RGBA
        return pil_image.fromarray(x.astype('uint8'), 'RGBA')
    elif x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])

def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=backend.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x

def load_from_pickle(filepath):
    file=open(filepath,'rb')
    content=pickle.load(file,encoding='bytes')
    file.close()
    return content

def fine_tune_filepreprocess1(featurepath):
    dirname = featurepath

    if os.path.exists(os.path.join(dirname,"train")):
        return
    else:
        os.makedirs(os.path.join(dirname,"train"))
        trainprefix=os.path.join(dirname,'train')
    file = open(os.path.join(dirname, 'train_batch'), 'rb')
    dict = pickle.load(file, encoding='bytes')
    print(dict.keys())
    train_x = dict[b'data']
    train_x = np.reshape(train_x, [50000, 3, 32, 32]).transpose([0, 2, 3, 1])
    train_y = dict[b'fine_labels']
    train_y = np.array(train_y)
    labelset=set(train_y)
    for label in labelset:
        os.makedirs(os.path.join(trainprefix,str(label)))
    for i in range(len(train_x)):
        misc.imsave(os.path.join(trainprefix,str(train_y[i]),"image_" + str(i) + ".jpg"),train_x[i])
        if i%1000==0:
            print (str(i)+"done")

    if os.path.exists(os.path.join(dirname,"test")):
        return
    else:
        os.makedirs(os.path.join(dirname,"test"))
        testprefix=os.path.join(dirname,'test')
    file = open(os.path.join(dirname, 'test_batch'), 'rb')
    dict = pickle.load(file, encoding='bytes')
    test_x = dict[b'data']
    test_x = np.reshape(test_x, [10000, 3, 32, 32]).transpose([0, 2, 3, 1])
    test_y = dict[b'fine_labels']
    test_y = np.array(test_y)
    labelset=set(test_y)
    for label in labelset:
        os.makedirs(os.path.join(testprefix, str(label)))
    for i in range(len(test_x)):
        misc.imsave(os.path.join(testprefix, str(test_y[i]), "image_" + str(i) + ".jpg"), test_x[i])
        if i % 1000 == 0:
            print(str(i) + "done")
    return

fine_tune_filepreprocess1('data/img_cls_data/cifar100')
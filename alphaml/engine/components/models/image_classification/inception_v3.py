from keras import layers, backend
from keras import Model
from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformIntegerHyperparameter

from alphaml.engine.components.models.base_dl_model import BaseImageClassificationModel
from alphaml.utils.constants import *


class Inceptionv3Classifier(BaseImageClassificationModel):
    def __init__(self, *args, **kwargs):
        self.batch_size = None
        self.keep_prob = None
        self.optimizer = None
        self.sgd_lr = None
        self.sgd_decay = None
        self.sgd_momentum = None
        self.adam_lr = None
        self.adam_decay = None
        self.inceptionv3_block_a = None
        self.inceptionv3_block_b = None
        self.inceptionv3_block_c = None
        self.estimator = None
        self.inputshape = None
        self.classnum = None
        self.default_size = 299
        self.work_size = 299
        self.min_size = 128
        self.model_name = 'InceptionV3'

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace()
        BaseImageClassificationModel.set_training_space(cs)
        BaseImageClassificationModel.set_optimizer_space(cs)
        inceptionv3_block_a = UniformIntegerHyperparameter('inceptionv3_block_a', 2, 4, default_value=3)
        inceptionv3_block_b = UniformIntegerHyperparameter('inceptionv3_block_b', 3, 5, default_value=4)
        inceptionv3_block_c = UniformIntegerHyperparameter('inceptionv3_block_c', 1, 3, default_value=2)
        cs.add_hyperparameters([inceptionv3_block_a, inceptionv3_block_b, inceptionv3_block_c])
        return cs

    @staticmethod
    def get_properties():
        return {'shortname': 'InceptionV3',
                'name': 'InceptionV4 Image Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'input': (DENSE),
                'output': (PREDICTIONS,)}

    def fit(self, data, **kwarg):
        self.validate_inputshape()
        self.load_data(data, **kwarg)
        self.base_model = Inception_v3(input_shape=self.inputshape,
                                       inceptionv3_block_a=self.inceptionv3_block_a,
                                       inceptionv3_block_b=self.inceptionv3_block_b,
                                       inceptionv3_block_c=self.inceptionv3_block_c)
        return super().fit(data, **kwarg)


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              strides=(1, 1),
              padding='same',
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x


def inception_a(x, blockid, pool_filters):
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, pool_filters, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed' + str(blockid))
    return x


def inception_b(x, blockid, bottleneck_filters):
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, bottleneck_filters, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, bottleneck_filters, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, bottleneck_filters, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, bottleneck_filters, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, bottleneck_filters, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, bottleneck_filters, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed' + str(blockid))
    return x


def inception_c(x, blockid):
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    branch1x1 = conv2d_bn(x, 320, 1, 1)

    branch3x3 = conv2d_bn(x, 384, 1, 1)
    branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
    branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)

    branch3x3dbl = conv2d_bn(x, 448, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
    branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
    branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)

    branch_pool = layers.AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch3x3_1, branch3x3_2, branch3x3dbl_1, branch3x3dbl_2, branch_pool],
        axis=channel_axis,
        name='mixed' + str(blockid))
    return x


def reduction_a(x, blockid):
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed' + str(blockid))
    return x


def reduction_b(x, blockid):
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='mixed' + str(blockid))
    return x


def Inception_v3(input_shape, **kwargs):
    kwargs = {k: kwargs[k] for k in kwargs if kwargs[k]}  # Remove None value in args

    img_input = layers.Input(shape=input_shape)

    # stem
    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = inception_a(x, 0, 32)
    index = 1
    for i in range(kwargs['inceptionv3_block_a'] - 1):
        x = inception_a(x, index, 64)
        index += 1

    x = reduction_a(x, index)
    index += 1

    x = inception_b(x, index, 128)
    index += 1
    for i in range(kwargs['inceptionv3_block_b'] - 2):
        x = inception_b(x, index, 160)
        index += 1
    x = inception_b(x, index, 192)
    index += 1

    x = reduction_b(x, index)
    index += 1
    for i in range(kwargs['inceptionv3_block_c']):
        x = inception_c(x, index)
        index += 1

    x = layers.GlobalAveragePooling2D()(x)
    # create model
    model = Model(inputs=img_input, outputs=x, name='InceptionV3')
    return model

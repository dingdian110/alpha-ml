from keras import layers, backend
from keras import Model
from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformIntegerHyperparameter

from alphaml.engine.components.models.base_dl_model import BaseImageClassificationModel
from alphaml.utils.constants import *


class Inceptionv4Classifier(BaseImageClassificationModel):
    def __init__(self, *args, **kwargs):
        self.batch_size = None
        self.keep_prob = None
        self.optimizer = None
        self.sgd_lr = None
        self.sgd_decay = None
        self.sgd_momentum = None
        self.adam_lr = None
        self.adam_decay = None
        self.inceptionv4_block_a = None
        self.inceptionv4_block_b = None
        self.inceptionv4_block_c = None
        self.estimator = None
        self.inputshape = None
        self.classnum = None
        self.default_size = 299
        self.work_size = 299
        self.min_size = 128
        self.model_name = 'InceptionV4'

    @staticmethod
    def get_properties():
        return {'shortname': 'InceptionV4',
                'name': 'InceptionV4 Image Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'input': (DENSE),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace()
        BaseImageClassificationModel.set_training_space(cs)
        BaseImageClassificationModel.set_optimizer_space(cs)
        inceptionv4_block_a = UniformIntegerHyperparameter('inceptionv4_block_a', 3, 5, default_value=4)
        inceptionv4_block_b = UniformIntegerHyperparameter('inceptionv4_block_b', 6, 8, default_value=7)
        inceptionv4_block_c = UniformIntegerHyperparameter('inceptionv4_block_c', 2, 4, default_value=3)
        cs.add_hyperparameters([inceptionv4_block_a, inceptionv4_block_b, inceptionv4_block_c])
        return cs

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, **kwarg):
        self.validate_inputshape()
        self.base_model = Inception_v4(self.inputshape,
                                       inceptionv4_block_a=self.inceptionv4_block_a,
                                       inceptionv4_block_b=self.inceptionv4_block_b,
                                       inceptionv4_block_c=self.inceptionv4_block_c)
        super().fit(x_train, y_train, x_valid, y_valid, **kwarg)


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


def stem(x_input):
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    x = conv2d_bn(x_input, 32, 3, 3, strides=(2, 2), padding='valid', name='stem_conv1')
    x = conv2d_bn(x, 32, 3, 3, padding='valid', name='stem_conv2')
    x = conv2d_bn(x, 64, 3, 3, name='stem_conv3')

    x1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='stem_pool1')(x)
    x2 = conv2d_bn(x, 96, 3, 3, strides=(2, 2), padding='valid', name='stem_conv4')

    x = layers.concatenate([x1, x2], axis=-1)

    x1 = conv2d_bn(x, 64, 1, 1, name='stem_conv5')
    x1 = conv2d_bn(x1, 96, 3, 3, padding='valid', name='stem_conv6')

    x2 = conv2d_bn(x, 64, 1, 1, name='stem_conv7')
    x2 = conv2d_bn(x2, 64, 7, 1, name='stem_conv8')
    x2 = conv2d_bn(x2, 64, 1, 7, name='stem_conv9')
    x2 = conv2d_bn(x2, 96, 3, 3, padding='valid', name='stem_conv10')

    x = layers.concatenate([x1, x2], axis=-1)

    x1 = conv2d_bn(x, 192, 3, 3, strides=(2, 2), padding='valid', name='stem_conv11')
    x2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='stem_pool2')(x)

    merged_vector = layers.concatenate([x1, x2], axis=channel_axis)
    return merged_vector


def inception_A(x, blockid):
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    nameprefix = 'inception_a' + str(blockid)
    branch_a = layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same',
                                       name=nameprefix + "_branch1_avgpool")(
        x)
    branch_a = conv2d_bn(branch_a, 96, 1, 1, name=nameprefix + "_branch1_conv1")

    branch_b = conv2d_bn(x, 96, 1, 1, name=nameprefix + "_branch2_conv1")

    branch_c = conv2d_bn(x, 64, 1, 1, name=nameprefix + "_branch3_conv1")
    branch_c = conv2d_bn(branch_c, 96, 3, 3, name=nameprefix + "_branch3_conv2")

    branch_d = conv2d_bn(x, 64, 1, 1, name=nameprefix + "_branch4_conv1")
    branch_d = conv2d_bn(branch_d, 96, 3, 3, name=nameprefix + "_branch4_conv2")
    branch_d = conv2d_bn(branch_d, 96, 3, 3, name=nameprefix + "_branch4_conv3")

    merged_vector = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=channel_axis)
    return merged_vector


def inception_B(x, blockid):
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    nameprefix = 'inception_b' + str(blockid)
    branch_a = layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same',
                                       name=nameprefix + "_branch1_avgpool")(x)
    branch_a = conv2d_bn(branch_a, 128, 1, 1, name=nameprefix + "_branch1_conv1")

    branch_b = conv2d_bn(x, 384, 1, 1, name="_branch2_conv1")

    branch_c = conv2d_bn(x, 192, 1, 1, name="_branch3_conv1")
    branch_c = conv2d_bn(branch_c, 224, 1, 7, name="_branch3_conv2")
    branch_c = conv2d_bn(branch_c, 256, 1, 7, name="_branch3_conv3")

    branch_d = conv2d_bn(x, 192, 1, 1, name=nameprefix + "_branch4_con1")
    branch_d = conv2d_bn(branch_d, 192, 1, 7, name=nameprefix + "_branch4_conv2")
    branch_d = conv2d_bn(branch_d, 224, 7, 1, name=nameprefix + "_branch4_conv3")
    branch_d = conv2d_bn(branch_d, 224, 1, 7, name=nameprefix + "_branch4_conv4")
    branch_d = conv2d_bn(branch_d, 256, 7, 1, name=nameprefix + "_branch4_conv5")

    merged_vector = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=channel_axis)
    return merged_vector


def inception_C(x, blockid):
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    nameprefix = 'inception_b' + str(blockid)
    branch_a = layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same',
                                       name=nameprefix + "_branch1_avgpool")(x)
    branch_a = conv2d_bn(branch_a, 256, 1, 1, name=nameprefix + "_branch1_conv1")

    branch_b = conv2d_bn(x, 256, 1, 1, name=nameprefix + "_branch2_conv1")

    branch_c = conv2d_bn(x, 384, 1, 1, name=nameprefix + "_branch3_conv1")
    branch_c1 = conv2d_bn(branch_c, 256, 1, 3, name=nameprefix + "_branch3_conv2")
    branch_c2 = conv2d_bn(branch_c, 256, 3, 1, name=nameprefix + "_branch3_conv3")

    branch_d = conv2d_bn(x, 384, 1, 1, name=nameprefix + "_branch4_conv1")
    branch_d = conv2d_bn(branch_d, 448, 1, 3, name=nameprefix + "_branch4_conv2")
    branch_d = conv2d_bn(branch_d, 512, 3, 1, name=nameprefix + "_branch4_conv3")
    branch_d1 = conv2d_bn(branch_d, 256, 3, 1, name=nameprefix + "_branch4_conv4")
    branch_d2 = conv2d_bn(branch_d, 256, 1, 3, name=nameprefix + "_branch4_conv5")

    merged_vector = layers.concatenate(
        [branch_a, branch_b, branch_c1, branch_c2, branch_d1, branch_d2], axis=channel_axis)
    return merged_vector


def reduction_A(x):
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    nameprefix = 'reduction_a'
    branch_a = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid',
                                   name=nameprefix + "_branch1_maxpool")(x)

    branch_b = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid', name=nameprefix + "_branch2_conv1")

    branch_c = conv2d_bn(x, 192, 1, 1, name=nameprefix + "_branch3_conv1")
    branch_c = conv2d_bn(branch_c, 224, 3, 3, name=nameprefix + "_branch3_conv2")
    branch_c = conv2d_bn(branch_c, 256, 3, 3, strides=(2, 2), padding='valid', name=nameprefix + "_branch3_conv3")

    merged_vector = layers.concatenate([branch_a, branch_b, branch_c], axis=channel_axis)
    return merged_vector


def reduction_B(x):
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    nameprefix = 'reduction_b'
    branch_a = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid',
                                   name=nameprefix + "_branch1_maxpool")(x)

    branch_b = conv2d_bn(x, 192, 1, 1, name=nameprefix + "_branch2_conv1")
    branch_b = conv2d_bn(branch_b, 192, 3, 3, strides=(2, 2), padding='valid', name=nameprefix + "_branch2_conv2")

    branch_c = conv2d_bn(x, 256, 1, 1, name=nameprefix + "_branch3_conv1")
    branch_c = conv2d_bn(branch_c, 256, 1, 7, name=nameprefix + "_branch3_conv2")
    branch_c = conv2d_bn(branch_c, 320, 7, 1, name=nameprefix + "_branch3_conv3")
    branch_c = conv2d_bn(branch_c, 320, 3, 3, strides=(2, 2), padding='valid', name=nameprefix + "_branch3_conv4")

    merged_vector = layers.concatenate([branch_a, branch_b, branch_c], axis=channel_axis)
    return merged_vector


def Inception_v4(input_shape, **kwargs):
    kwargs = {k: kwargs[k] for k in kwargs if kwargs[k]}  # Remove None value in args

    img_input = layers.Input(shape=input_shape)

    x = stem(img_input)

    for i in range(1, kwargs['inceptionv4_block_a'] + 1):
        x = inception_A(x, i)

    x = reduction_A(x)

    for i in range(1, kwargs['inceptionv4_block_b'] + 1):
        x = inception_B(x, i)

    x = reduction_B(x)

    for i in range(1, kwargs['inceptionv4_block_c'] + 1):
        x = inception_C(x, i)

    x = layers.GlobalAveragePooling2D()(x)
    # create model
    model = Model(inputs=img_input, outputs=x, name='InceptionV4')
    model.summary()
    return model

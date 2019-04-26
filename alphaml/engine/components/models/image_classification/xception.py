import numpy as np
from keras import layers
from keras import Model
from keras import backend
from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformIntegerHyperparameter, CategoricalHyperparameter

from alphaml.engine.components.models.base_dl_model import BaseImageClassificationModel
from alphaml.utils.constants import *


class XceptionClassifier(BaseImageClassificationModel):
    def __init__(self):
        self.batch_size = None
        self.keep_prob = None
        self.optimizer = None
        self.sgd_lr = None
        self.sgd_decay = None
        self.sgd_momentum = None
        self.adam_lr = None
        self.adam_decay = None
        self.xception_middle_blocks = None
        self.estimator = None
        self.inputshape = None
        self.classnum = None
        self.min_size = 32
        self.work_size = 299
        self.default_size = 299
        self.model_name = 'Xception'

    @staticmethod
    def get_properties():
        return {'shortname': 'Xception',
                'name': 'Xception Image Classifier',
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
        xception_middle_blocks = UniformIntegerHyperparameter('xception_middle_blocks', 6, 10, default_value=8)
        cs.add_hyperparameter(xception_middle_blocks)
        return cs

    def fit(self, data, **kwarg):
        self.validate_inputshape()
        self.load_data(data, **kwarg)
        self.base_model = Xception(self.inputshape,
                                   xception_middle_blocks=self.xception_middle_blocks)
        super().fit(data, **kwarg)


def Xception(input_shape, **kwargs):
    kwargs = {k: kwargs[k] for k in kwargs if kwargs[k]}  # Remove None value in args

    img_input = layers.Input(shape=input_shape)
    if backend.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    x = layers.Conv2D(32, (3, 3),
                      strides=(2, 2),
                      use_bias=False,
                      name='block1_conv1')(img_input)
    x = layers.BatchNormalization(name='block1_conv1_bn', axis=bn_axis)(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)
    x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = layers.BatchNormalization(name='block1_conv2_bn', axis=bn_axis)(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)

    residual = layers.Conv2D(128, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization(axis=bn_axis)(residual)

    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv1')(x)
    x = layers.BatchNormalization(name='block2_sepconv1_bn', axis=bn_axis)(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv2')(x)
    x = layers.BatchNormalization(name='block2_sepconv2_bn', axis=bn_axis)(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=bn_axis)(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv1')(x)
    x = layers.BatchNormalization(name='block3_sepconv1_bn', axis=bn_axis)(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv2')(x)
    x = layers.BatchNormalization(name='block3_sepconv2_bn', axis=bn_axis)(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(728, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization(axis=bn_axis)(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv1')(x)
    x = layers.BatchNormalization(name='block4_sepconv1_bn', axis=bn_axis)(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv2')(x)
    x = layers.BatchNormalization(name='block4_sepconv2_bn', axis=bn_axis)(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(kwargs['xception_middle_blocks']):
        residual = x
        prefix = 'block' + str(i + 5)

        x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv1')(x)
        x = layers.BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv2')(x)
        x = layers.BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv3')(x)
        x = layers.BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    blockcnt = 5 + kwargs['xception_middle_blocks']
    prefix = 'block' + str(blockcnt)
    residual = layers.Conv2D(1024, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=bn_axis)(residual)

    x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name=prefix + '_sepconv1')(x)
    x = layers.BatchNormalization(name=prefix + '_sepconv1_bn')(x)
    x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
    x = layers.SeparableConv2D(1024, (3, 3),
                               padding='same',
                               use_bias=False,
                               name=prefix + '_sepconv2')(x)
    x = layers.BatchNormalization(name=prefix + '_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name=prefix + '_pool')(x)
    x = layers.add([x, residual])

    blockcnt += 1
    prefix = 'block' + str(blockcnt)
    x = layers.SeparableConv2D(1536, (3, 3),
                               padding='same',
                               use_bias=False,
                               name=prefix + '_sepconv1')(x)
    x = layers.BatchNormalization(name=prefix + '_sepconv1_bn')(x)
    x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)

    x = layers.SeparableConv2D(2048, (3, 3),
                               padding='same',
                               use_bias=False,
                               name=prefix + '_sepconv2')(x)
    x = layers.BatchNormalization(name=prefix + '_sepconv2_bn')(x)
    x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)

    x = layers.GlobalAveragePooling2D()(x)
    model = Model(img_input, x, name='xception')
    return model

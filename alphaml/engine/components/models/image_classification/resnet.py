import numpy as np
from keras import layers
from keras import Model
from keras import backend
from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformIntegerHyperparameter, CategoricalHyperparameter

from alphaml.engine.components.models.base_dl_model import BaseImageClassificationModel
from alphaml.utils.constants import *


class ResNetClassifier(BaseImageClassificationModel):
    def __init__(self, *arg, **karg):
        self.batch_size = None
        self.keep_prob = None
        self.optimizer = None
        self.sgd_lr = None
        self.sgd_decay = None
        self.sgd_momentum = None
        self.adam_lr = None
        self.adam_decay = None
        self.res_kernel_size = None
        self.res_stage2_block = None
        self.res_stage3_block = None
        self.res_stage4_block = None
        self.res_stage5_block = None
        self.estimator = None
        self.inputshape = None
        self.classnum = None
        self.min_size = 32
        self.work_size = 197
        self.default_size = 224
        self.model_name = 'ResNet'

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'ResNet',
                'name': 'ResNet Image Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'input': (DENSE),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        BaseImageClassificationModel.set_training_space(cs)
        BaseImageClassificationModel.set_optimizer_space(cs)
        res_kernel_size = CategoricalHyperparameter('res_kernel_size', [3, 5], default_value=3)
        res_stage2_block = UniformIntegerHyperparameter('res_stage2_block', 1, 3, default_value=2)
        res_stage3_block = UniformIntegerHyperparameter('res_stage3_block', 1, 11, default_value=3)
        res_stage4_block = UniformIntegerHyperparameter('res_stage4_block', 1, 47, default_value=5)
        res_stage5_block = UniformIntegerHyperparameter('res_stage5_block', 1, 3, default_value=2)
        cs.add_hyperparameters(
            [res_kernel_size, res_stage2_block, res_stage3_block, res_stage4_block, res_stage5_block])
        return cs

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, **kwarg):
        self.validate_inputshape()
        self.base_model = ResNet(input_shape=self.inputshape,
                                 res_kernel_size=self.res_kernel_size,
                                 res_stage2_block=self.res_stage2_block,
                                 res_stage3_block=self.res_stage3_block,
                                 res_stage4_block=self.res_stage4_block,
                                 res_stage5_block=self.res_stage5_block)
        super().fit(x_train, y_train, x_valid, y_valid, **kwarg)


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters,
               stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet(input_shape, **kwargs):
    """Instantiates the ResNet architecture.

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.

    # Returns
        A Keras model instance.

    # ResNet configuration space:
        kernel_size: 3,5
        stage2_block: [1,3]
        stage3_block: [1,11]
        stage4_block: [1,47]
        stage5_block: [1,4]
    """

    kwargs = {k: kwargs[k] for k in kwargs if kwargs[k]}  # Remove None value in args

    kernel_size = kwargs['res_kernel_size']
    stages = 4

    img_input = layers.Input(shape=input_shape)
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    filters = 64

    # stage 1
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(filters, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # stage 2-5
    for stage in range(2, stages + 2):
        if stage == 2:
            x = conv_block(x, kernel_size, [filters, filters, filters * 4], stage=stage, block='_0_', strides=(1, 1))
        else:
            x = conv_block(x, kernel_size, [filters, filters, filters * 4], stage=stage, block='_0_')
        for i in range(kwargs['res_stage' + str(stage) + '_block']):
            x = identity_block(x, 3, [filters, filters, filters * 4], stage=stage, block="_" + str(i + 1) + "_")
        filters *= 2

    x = layers.GlobalAveragePooling2D()(x)
    # Create model.
    model = Model(img_input, x, name='resnet')
    return model

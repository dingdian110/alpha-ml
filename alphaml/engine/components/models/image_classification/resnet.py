from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import keras
from keras_applications import get_keras_submodule
from keras import Model
from keras.layers import Dropout, Dense
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    InCondition
from ..base_model import BaseClassificationModel
from ...data_preprocessing.image_preprocess import preprocess

backend = get_keras_submodule('backend')
engine = get_keras_submodule('engine')
layers = get_keras_submodule('layers')
models = get_keras_submodule('models')
keras_utils = get_keras_submodule('utils')


class ResNetClassifier(BaseClassificationModel):
    def __init__(self, batch_size, keep_prob, optimizer,
                 sgd_lr, sgd_decay, sgd_momentum,
                 adam_lr, adam_decay,
                 res_kernel_size, res_stage2_block,
                 res_stage3_block, res_stage4_block,
                 res_stage5_block, **arg):
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.optimizer = optimizer
        self.sgd_lr = sgd_lr
        self.sgd_decay = sgd_decay
        self.sgd_momentum = sgd_momentum
        self.adam_lr = adam_lr
        self.adam_decay = adam_decay
        self.res_kernel_size = res_kernel_size
        self.res_stage2_block = res_stage2_block
        self.res_stage3_block = res_stage3_block
        self.res_stage4_block = res_stage4_block
        self.res_stage5_block = res_stage5_block
        self.estimator = None

    def get_hyperparameter_search_space(self):
        cs = ConfigurationSpace()
        # base config
        batch_size = CategoricalHyperparameter('batch_size', [16, 32], default_value=32)
        keep_prob = UniformFloatHyperparameter('keep_prob', 0, 0.99, default_value=0.5)

        # optimizer config
        optimizer = CategoricalHyperparameter('optimizer', ['SGD', 'Adam'], default_value='Adam')
        sgd_lr = UniformFloatHyperparameter('sgd_lr', 0.00001, 0.1,
                                            default_value=0.005, log=True)  # log scale
        sgd_decay = UniformFloatHyperparameter('sgd_decay', 0.0001, 0.1,
                                               default_value=0.05, log=True)  # log scale
        sgd_momentum = UniformFloatHyperparameter('sgd_momentum', 0.3, 0.99, default_value=0.9)
        adam_lr = UniformFloatHyperparameter('adam_lr', 0.00001, 0.1,
                                             default_value=0.005, log=True)  # log scale
        adam_decay = UniformFloatHyperparameter('adam_decay', 0.0001, 0.1,
                                                default_value=0.05, log=True)  # log scale

        sgd_lr_cond = InCondition(child=sgd_lr, parent=optimizer, values=['SGD'])
        sgd_decay_cond = InCondition(child=sgd_decay, parent=optimizer, values=['SGD'])
        sgd_momentum_cond = InCondition(child=sgd_momentum, parent=optimizer, values=['SGD'])
        adam_lr_cond = InCondition(child=adam_lr, parent=optimizer, values=['Adam'])
        adam_decay_cond = InCondition(child=adam_decay, parent=optimizer, values=['Adam'])

        # network config
        res_kernel_size = CategoricalHyperparameter('res_kernel_size', [3, 5], default_value=3)
        res_stage2_block = UniformIntegerHyperparameter('res_stage2_block', 1, 3, default_value=2)
        res_stage3_block = UniformIntegerHyperparameter('res_stage3_block', 1, 11, default_value=3)
        res_stage4_block = UniformIntegerHyperparameter('res_stage4_block', 1, 47, default_value=5)
        res_stage5_block = UniformIntegerHyperparameter('res_stage5_block', 1, 3, default_value=2)

        cs.add_hyperparameters([batch_size, keep_prob,
                                optimizer, sgd_lr, sgd_decay, sgd_momentum, adam_lr, adam_decay,
                                res_kernel_size, res_stage2_block, res_stage3_block, res_stage4_block,
                                res_stage5_block])
        cs.add_conditions([sgd_lr_cond, sgd_decay_cond, sgd_momentum_cond, adam_lr_cond, adam_decay_cond])
        return cs

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, **karg):
        timestr = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
        if x_valid is None and y_valid is None:
            if_valid = False
        else:
            if_valid = True

        if self.optimizer == 'SGD':
            optimizer = SGD(self.sgd_lr, self.sgd_momentum, self.sgd_decay)
        elif self.optimizer == 'Adam':
            optimizer = Adam(self.adam_lr, decay=self.adam_decay)
        else:
            raise ValueError('No optimizer named %s defined' % str(self.optimizer))

        trainpregen, validpregen, _ = preprocess()
        train_gen = trainpregen.flow(x_train, y_train, batch_size=self.batch_size)
        if if_valid:
            valid_gen = validpregen.flow(x_valid, y_valid, batch_size=self.batch_size)

        # remain to get
        inputshape = (32, 32, 3)
        classnum = 10
        # model
        base_model = ResNet(input_shape=inputshape,
                            res_kernel_size=self.res_kernel_size,
                            res_stage2_block=self.res_stage2_block,
                            res_stage3_block=self.res_stage3_block,
                            res_stage4_block=self.res_stage4_block,
                            res_stage5_block=self.res_stage5_block)
        y = base_model.output
        y = Dropout(1 - self.keep_prob)(y)
        y = Dense(classnum)(y)
        model = Model(inputs=base_model.input, outputs=y)
        checkpoint = ModelCheckpoint(filepath='model_%s.hdf5' % timestr,
                                     monitor='val_acc',
                                     save_best_only=True,
                                     period=1)
        earlystop = EarlyStopping(monitor='val_acc', patience=8)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
        model.fit_generator(generator=train_gen,
                            epochs=120,
                            validation_data=valid_gen,
                            callbacks=[checkpoint, earlystop])
        final_result = checkpoint.best
        return self  # minimize validation accuracy


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


def ResNet(input_shape, **args):
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
        kernel_size: 3,5,7
        stage2_block: [1,3]
        stage3_block: [1,11]
        stage4_block: [1,47]
        stage5_block: [1,4]
    """

    args = {k: args[k] for k in args if args[k]}  # Remove None value in args

    assert isinstance(args, dict)
    kernel_size = args['res_kernel_size']
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
        x = conv_block(x, kernel_size, [filters, filters, filters * 4], stage=stage, block='_0_', strides=(1, 1))
        for i in range(args['res_stage' + str(stage) + '_block']):
            x = identity_block(x, 3, [filters, filters, filters * 4], stage=stage, block="_" + str(i + 1) + "_")
        filters *= 2

    x = layers.GlobalAveragePooling2D()(x)
    # Create model.
    model = models.Model(img_input, x, name='resnet')
    return model

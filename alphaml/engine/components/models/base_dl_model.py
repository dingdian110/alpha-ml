from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import warnings
from keras import Model
from keras import layers
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformFloatHyperparameter, CategoricalHyperparameter, InCondition
from alphaml.engine.components.models.base_model import BaseClassificationModel
from alphaml.engine.components.data_preprocessing.image_preprocess import preprocess


class BaseImageClassificationModel(BaseClassificationModel):
    def __init__(self):
        self.base_model = None
        self.min_size = None
        self.default_size = None
        super().__init__()

    def set_model_config(self, inputshape, classnum, *args, **kwargs):
        self.inputshape = inputshape
        self.classnum = classnum

    @staticmethod
    def set_training_space(cs: ConfigurationSpace):
        batch_size = CategoricalHyperparameter('batch_size', [16, 32], default_value=32)
        keep_prob = UniformFloatHyperparameter('keep_prob', 0, 0.99, default_value=0.5)
        cs.add_hyperparameters([batch_size, keep_prob])

    @staticmethod
    def set_optimizer_space(cs: ConfigurationSpace):
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

        cs.add_hyperparameters([optimizer, sgd_lr, sgd_decay, sgd_momentum, adam_lr, adam_decay])
        cs.add_conditions([sgd_lr_cond, sgd_decay_cond, sgd_momentum_cond, adam_lr_cond, adam_decay_cond])

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, sample_weight=None):
        if self.base_model is None:
            raise AttributeError("Base model is not defined!")

        if self.inputshape[0] < self.min_size or self.inputshape[1] < self.min_size:
            warnings.warn("The minimum inputshape of the model is " + str((self.min_size, self.min_size)) +
                          ", while " + str(self.inputshape[0:2]) + " given.")

        if self.optimizer == 'SGD':
            optimizer = SGD(self.sgd_lr, self.sgd_momentum, self.sgd_decay)
        elif self.optimizer == 'Adam':
            optimizer = Adam(self.adam_lr, decay=self.adam_decay)
        else:
            raise ValueError('No optimizer named %s defined' % str(self.optimizer))

        timestr = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
        if x_valid is None and y_valid is None:
            if_valid = False
        else:
            if_valid = True

        trainpregen, validpregen = preprocess()
        train_gen = trainpregen.flow(x_train, y_train, batch_size=self.batch_size)
        if if_valid:
            valid_gen = validpregen.flow(x_valid, y_valid, batch_size=self.batch_size)
            checkpoint_monitor = 'val_acc'
        else:
            valid_gen = None
            checkpoint_monitor = 'acc'

        # model
        if self.classnum == 1:
            final_activation = 'sigmoid'
            loss = 'binary_crossentropy'
        else:
            final_activation = 'softmax'
            loss = 'categorical_crossentropy'

        y = self.base_model.output
        y = layers.Dropout(1 - self.keep_prob)(y)
        y = layers.Dense(self.classnum, activation=final_activation, name='Dense_final')(y)
        model = Model(inputs=self.base_model.input, outputs=y)

        # TODO: load models after training
        checkpoint = ModelCheckpoint(filepath='model_%s.hdf5' % timestr,
                                     monitor=checkpoint_monitor,
                                     save_best_only=True,
                                     period=1)
        earlystop = EarlyStopping(monitor='val_acc', patience=8)
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
        model.fit_generator(generator=train_gen,
                            epochs=200,
                            validation_data=valid_gen,
                            callbacks=[checkpoint, earlystop])
        self.estimator = model
        self.best_result = checkpoint.best
        return self

    def fit_from_directory(self, dirname, sample_weight=None):
        pass

from keras import layers
from keras import Model
from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformIntegerHyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter

from alphaml.engine.components.models.base_dl_model import BaseImageClassificationModel
from alphaml.utils.constants import *


class VGGNetClassifier(BaseImageClassificationModel):
    def __init__(self, *args, **kwargs):
        self.batch_size = None
        self.keep_prob = None
        self.optimizer = None
        self.sgd_lr = None
        self.sgd_decay = None
        self.sgd_momentum = None
        self.adam_lr = None
        self.adam_decay = None
        self.vgg_kernel_size = None
        self.vgg_keep_prob = None
        self.vgg_block2_layer = None
        self.vgg_block3_layer = None
        self.vgg_block4_layer = None
        self.vgg_block5_layer = None
        self.estimator = None
        self.inputshape = None
        self.classnum = None
        self.min_size = 32
        self.work_size = 48
        self.default_size = 224
        self.model_name = 'VGGNet'

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'VGGNet',
                'name': 'VGGNet Image Classifier',
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
        vgg_kernel_size = CategoricalHyperparameter('vgg_kernel_size', [3, 5], default_value=3)
        vgg_keep_prob = UniformFloatHyperparameter('vgg_keep_prob', 0, 0.99, default_value=0.5)
        vgg_block2_layer = UniformIntegerHyperparameter('vgg_block2_layer', 2, 3, default_value=2)
        vgg_block3_layer = UniformIntegerHyperparameter('vgg_block3_layer', 2, 5, default_value=3)
        vgg_block4_layer = UniformIntegerHyperparameter('vgg_block4_layer', 2, 5, default_value=3)
        vgg_block5_layer = UniformIntegerHyperparameter('vgg_block5_layer', 2, 5, default_value=3)
        cs.add_hyperparameters(
            [vgg_kernel_size, vgg_keep_prob, vgg_block2_layer, vgg_block3_layer, vgg_block4_layer, vgg_block5_layer])
        return cs

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, **kwarg):
        self.validate_inputshape()
        self.base_model = VGGNet(input_shape=self.inputshape,
                                 vgg_kernel_size=self.vgg_kernel_size,
                                 vgg_keep_prob=self.keep_prob,
                                 vgg_block2_layer=self.vgg_block2_layer,
                                 vgg_block3_layer=self.vgg_block3_layer,
                                 vgg_block4_layer=self.vgg_block4_layer,
                                 vgg_block5_layer=self.vgg_block5_layer)
        super().fit(x_train, y_train, x_valid, y_valid, **kwarg)


def VGGNet(input_shape, **kwargs):
    """Instantiates the VGG architecture.

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.

    # Returns
        A Keras model instance.

    # VGGNet configuration space:
        kernel_size: [3,5]
    """
    kwargs = {k: kwargs[k] for k in kwargs if kwargs[k]}  # Remove None value in args

    img_input = layers.Input(shape=input_shape)
    blocks = 5
    block_layers = [2, kwargs['vgg_block2_layer'], kwargs['vgg_block3_layer'], kwargs['vgg_block4_layer'],
                    kwargs['vgg_block5_layer']]
    block_filters = [64, 128, 256, 512, 512]
    kernel_size = kwargs['vgg_kernel_size']
    keep_prob = kwargs['vgg_keep_prob']
    for i in range(blocks):
        for j in range(block_layers[i]):
            if i == 0 and j == 0:
                x = layers.Conv2D(block_filters[i], kernel_size, activation='relu', padding='same',
                                  name='block' + str(i + 1) + "_conv" + str(j + 1))(img_input)
            else:
                x = layers.Conv2D(block_filters[i], kernel_size, activation='relu', padding='same',
                                  name='block' + str(i + 1) + "_conv" + str(j + 1))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block' + str(i + 1) + "_pool")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dropout(1 - keep_prob)(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)
    model = Model(img_input, x, name='vggnet')
    return model

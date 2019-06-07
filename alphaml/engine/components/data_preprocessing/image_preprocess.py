from keras.preprocessing.image import ImageDataGenerator
from alphaml.engine.components.data_manager import DataManager


def preprocess(data: DataManager, config={}):
    assert isinstance(config, dict)
    rotation_range = config.get('rotation_range', 15)
    width_shift_range = config.get('width_shift_range', 0.2)
    height_shift_range = config.get('height_shift_range', 0.2)
    horizontal_flip = config.get('horizontal_flip', True)
    train_datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        rescale=1.0 / 255,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        horizontal_flip=horizontal_flip,
        validation_split=data.split_size)  # Default data augmentation
    valid_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )
    return train_datagen, valid_datagen

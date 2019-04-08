from keras.preprocessing.image import ImageDataGenerator

def preprocess(if_train=True,config={}):
    assert isinstance(config,dict)
    rotation_range= config.get('rotation_range',15)
    width_shift_range=config.get('width_shift_range',0.2)
    height_shift_range=config.get('height_shift_range',0.2)
    horizontal_flip=config.get('horizontal_flip',True)
    if if_train:
        train_datagen = ImageDataGenerator(
            rotation_range=rotation_range,
            rescale=1.0 / 255,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            horizontal_flip=horizontal_flip)  # Default data augmentation
    else:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255
        )
    valid_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )
    return train_datagen,valid_datagen
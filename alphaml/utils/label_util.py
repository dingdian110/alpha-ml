import numpy as np


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def map_label(y, map_dict={}, if_binary=False):
    """Convert a class vector (all types) to a class vector (integers)

    E.g. for use with to_categorical. ['a', 'b', 4, 7, 'a'] -> [0, 1, 2, 3, 0],
        ['a', 'b', 'a', 'a'] -> [0, 1, 0, 0]

    # Arguments
        y: class vector to be converted
        map_dict: mapping relations

    # Returns:
        A converted class vector and two dictionaries of mapping relations.
    """

    assert isinstance(map_dict, dict)
    y = np.array(y)
    y = y.ravel()
    if not map_dict:
        if_validate = False
    else:
        if if_binary and len(map_dict) != 2:
            raise ValueError(
                "Expected a dictionary of 2 elements in map_dict while received %d elements!" % len(map_dict))
        if_validate = True
    rev_map_dict = {}
    class_idx = 0
    int_y = []
    for label_element in y:
        if label_element not in map_dict:
            if if_validate:
                raise ValueError("Invalid label %s!" % str(label_element))
            map_dict[label_element] = class_idx
            rev_map_dict[class_idx] = label_element
            class_idx += 1
            if if_binary and class_idx > 1:
                raise ValueError("Found more than 2 classes in label inputs!")
        int_y.append(map_dict[label_element])
    int_y = np.array(int_y, dtype='int')
    return int_y, map_dict, rev_map_dict


def get_classnum(y):
    """Get classnum from one-hot label inputs 'y'. Note that this function will not validate the label inputs

    # Arguments
        y: label inputs

    # Returns:
        The number of classes in 'y'
    """
    assert isinstance(y, np.ndarray)
    inputshape = y.shape
    if len(inputshape) == 2:
        return inputshape[-1]
    else:
        raise ValueError("Input labels should be a 2-dim one-hot vector!")

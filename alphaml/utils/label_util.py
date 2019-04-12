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


def map_label(y, map_dict=None):
    """Convert a class vector (all types) to a class vector (integers)

    E.g. for use with to_categorical. ['a', 'b', 4, 7, 'a'] -> [0, 1, 2, 3, 0]

    # Arguments
        y: class vector to be converted
        map_dict: mapping relations

    # Returns:
        A converted class vector and two dictionaries of mapping relations.
    """

    y = np.array(y)
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if map_dict is None:
        if_validate = False
        map_dict = {}
    else:
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
        int_y.append(map_dict[label_element])
    int_y = np.array(int_y, dtype='int')
    return int_y, map_dict, rev_map_dict

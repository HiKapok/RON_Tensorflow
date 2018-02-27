import tensorflow as tf

__weights_dict = dict()

is_train = True

def load_weights(weight_file):
    import numpy as np

    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


def KitModel(weight_file = None):
    global __weights_dict
    __weights_dict = load_weights(weight_file)

    input           = tf.placeholder(tf.float32,  shape = (None, 224, 224, 3), name = 'input')
    ConvNd_0_pad    = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT', name='ConvNd_0_pad')
    ConvNd_0        = convolution(ConvNd_0_pad, group=1, strides=[1, 1], padding='VALID', name='ConvNd_0')
    Threshold_1     = tf.nn.relu(ConvNd_0, name = 'Threshold_1')
    ConvNd_2_pad    = tf.pad(Threshold_1, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT', name='ConvNd_2_pad')
    ConvNd_2        = convolution(ConvNd_2_pad, group=1, strides=[1, 1], padding='VALID', name='ConvNd_2')
    Threshold_3     = tf.nn.relu(ConvNd_2, name = 'Threshold_3')
    MaxPool2d_4     = tf.nn.max_pool(Threshold_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='MaxPool2d_4')
    ConvNd_5_pad    = tf.pad(MaxPool2d_4, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT', name='ConvNd_5_pad')
    ConvNd_5        = convolution(ConvNd_5_pad, group=1, strides=[1, 1], padding='VALID', name='ConvNd_5')
    Threshold_6     = tf.nn.relu(ConvNd_5, name = 'Threshold_6')
    ConvNd_7_pad    = tf.pad(Threshold_6, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT', name='ConvNd_7_pad')
    ConvNd_7        = convolution(ConvNd_7_pad, group=1, strides=[1, 1], padding='VALID', name='ConvNd_7')
    Threshold_8     = tf.nn.relu(ConvNd_7, name = 'Threshold_8')
    MaxPool2d_9     = tf.nn.max_pool(Threshold_8, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='MaxPool2d_9')
    ConvNd_10_pad   = tf.pad(MaxPool2d_9, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT', name='ConvNd_10_pad')
    ConvNd_10       = convolution(ConvNd_10_pad, group=1, strides=[1, 1], padding='VALID', name='ConvNd_10')
    Threshold_11    = tf.nn.relu(ConvNd_10, name = 'Threshold_11')
    ConvNd_12_pad   = tf.pad(Threshold_11, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT', name='ConvNd_12_pad')
    ConvNd_12       = convolution(ConvNd_12_pad, group=1, strides=[1, 1], padding='VALID', name='ConvNd_12')
    Threshold_13    = tf.nn.relu(ConvNd_12, name = 'Threshold_13')
    ConvNd_14_pad   = tf.pad(Threshold_13, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT', name='ConvNd_14_pad')
    ConvNd_14       = convolution(ConvNd_14_pad, group=1, strides=[1, 1], padding='VALID', name='ConvNd_14')
    Threshold_15    = tf.nn.relu(ConvNd_14, name = 'Threshold_15')
    MaxPool2d_16    = tf.nn.max_pool(Threshold_15, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='MaxPool2d_16')
    ConvNd_17_pad   = tf.pad(MaxPool2d_16, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT', name='ConvNd_17_pad')
    ConvNd_17       = convolution(ConvNd_17_pad, group=1, strides=[1, 1], padding='VALID', name='ConvNd_17')
    Threshold_18    = tf.nn.relu(ConvNd_17, name = 'Threshold_18')
    ConvNd_19_pad   = tf.pad(Threshold_18, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT', name='ConvNd_19_pad')
    ConvNd_19       = convolution(ConvNd_19_pad, group=1, strides=[1, 1], padding='VALID', name='ConvNd_19')
    Threshold_20    = tf.nn.relu(ConvNd_19, name = 'Threshold_20')
    ConvNd_21_pad   = tf.pad(Threshold_20, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT', name='ConvNd_21_pad')
    ConvNd_21       = convolution(ConvNd_21_pad, group=1, strides=[1, 1], padding='VALID', name='ConvNd_21')
    Threshold_22    = tf.nn.relu(ConvNd_21, name = 'Threshold_22')
    MaxPool2d_23    = tf.nn.max_pool(Threshold_22, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='MaxPool2d_23')
    ConvNd_24_pad   = tf.pad(MaxPool2d_23, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT', name='ConvNd_24_pad')
    ConvNd_24       = convolution(ConvNd_24_pad, group=1, strides=[1, 1], padding='VALID', name='ConvNd_24')
    Threshold_25    = tf.nn.relu(ConvNd_24, name = 'Threshold_25')
    ConvNd_26_pad   = tf.pad(Threshold_25, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT', name='ConvNd_26_pad')
    ConvNd_26       = convolution(ConvNd_26_pad, group=1, strides=[1, 1], padding='VALID', name='ConvNd_26')
    Threshold_27    = tf.nn.relu(ConvNd_26, name = 'Threshold_27')
    ConvNd_28_pad   = tf.pad(Threshold_27, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT', name='ConvNd_28_pad')
    ConvNd_28       = convolution(ConvNd_28_pad, group=1, strides=[1, 1], padding='VALID', name='ConvNd_28')
    Threshold_29    = tf.nn.relu(ConvNd_28, name = 'Threshold_29')
    MaxPool2d_30_pad = tf.pad(Threshold_29, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT', name='MaxPool2d_30_pad')
    MaxPool2d_30    = tf.nn.max_pool(MaxPool2d_30_pad, [1, 3, 3, 1], [1, 1, 1, 1], padding='VALID', name='MaxPool2d_30')
    ConvNd_31_pad   = tf.pad(MaxPool2d_30, [[0, 0], [6, 6], [6, 6], [0, 0]], 'CONSTANT', name='ConvNd_31_pad')
    ConvNd_31       = convolution(ConvNd_31_pad, group=1, strides=[1, 1], padding='VALID', name='ConvNd_31')
    Threshold_32    = tf.nn.relu(ConvNd_31, name = 'Threshold_32')
    ConvNd_33       = convolution(Threshold_32, group=1, strides=[1, 1], padding='VALID', name='ConvNd_33')
    Threshold_34    = tf.nn.relu(ConvNd_33, name = 'Threshold_34')
    return input, Threshold_34


def convolution(input, name, group, **kwargs):
    w = tf.Variable(__weights_dict[name]['weights'], trainable=is_train, name=name + "_weight")
    if group == 1:
        layer = tf.nn.convolution(input, w, **kwargs)
    else:
        weight_groups = tf.split(w, num_or_size_splits=group, axis=-1)
        xs = tf.split(input, num_or_size_splits=group, axis=-1)
        convolved = [tf.nn.convolution(x, weight, **kwargs) for
                    (x, weight) in zip(xs, weight_groups)]
        layer = tf.concat(convolved, axis=-1)

    if 'bias' in __weights_dict[name]:
        b = tf.Variable(__weights_dict[name]['bias'], trainable=is_train, name=name + "_bias")
        layer = layer + b
    return layer

import tensorflow as tf
from functools import reduce


def y_ordinal_to_y(y_ord, num_classes):
    """
    Converts ordinal encoded y to regular y
    where:
    y_ord = [> y_1,   > y_1, y_2,   > y_1, y_2, y_3]
    e.g.
    y_ord = [0.8, 0.1, 0.0]
    y_reg = [1.0-0.8=0.2, 0.8-0.1=0.7, 0.1-0.0=0.1, 0.0]
          = [0.2, 0.7, 0.1, 0.0]
    See paper for details: http://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf
    """
    # with tf.device('/cpu:0'):
    y_ord_shape = y_ord.get_shape().as_list()
    # if y_ord_shape[-1] == 1:
        # Must convert to ohe
    # y_ord = tf.one_hot(tf.cast(y_ord, tf.int32), axis=-1, depth=num_classes - 1)
    # if len(y_ord_shape) > 2:
    N = reduce(lambda x, y: x * y, y_ord_shape[:-1])
    y_ord = tf.reshape(y_ord, [N, num_classes - 1])

    y_filter = tf.constant([[1], [-1]], dtype=tf.float32)

    # pad LHS with ones, RHS with zeros
    y_ord_pad_lhs = tf.pad(y_ord - 1, tf.constant([[0, 0], [1, 0]], dtype='int32')) + 1
    y_ord_pad = tf.pad(y_ord_pad_lhs, tf.constant([[0, 0], [0, 1]], dtype='int32'))

    # add an extra channel dimension to allow for convolution
    y_ord_pad_wch = tf.expand_dims(y_ord_pad, -1)
    y_filter_wch = tf.expand_dims(y_filter, -1)

    y_reg = tf.squeeze(tf.nn.conv1d(y_ord_pad_wch, y_filter_wch, stride=1, padding='VALID'), [2])

    # this normalization is used to stop craziness at the start of training when y_ord might not respect ordering.
    # it is a conditional translation to have all positive values followed by normalization to ensure 0<=y<=1

    # def conditional_shift(y):
    #     cond = tf.greater_equal(tf.reduce_sum(tf.cast(tf.less(y, 0.0), tf.float32), axis=0), 1.0)
    #     return tf.cond(cond, lambda: y + tf.abs(tf.reduce_min(y, axis=0)), lambda: y)
    #
    # y_reg_shifted = tf.map_fn(conditional_shift, y_reg)
    # normalization_constant_matrix = tf.tile(tf.reshape(1.0 / tf.reduce_sum(y_reg_shifted, axis=1), [-1, 1]),
    #                                         [1, y_reg_shifted.get_shape().as_list()[1]])
    # y_reg_normalized = tf.multiply(normalization_constant_matrix, y_reg_shifted)
    # y_reg_normalized_reshaped = tf.reshape(y_reg_normalized, y_ord_shape[:-1] + [num_classes])
    return tf.reshape(y_reg, y_ord_shape[:-1] + [num_classes])


def y_to_y_ordinal(y_reg, num_classes=None):
    """
    Converts regular y to ordinal encoded y
    Assumes y_reg is a matrix of one-hot vectors
    e.g.
    y = [0, 0, 1, 0
         0, 1, 0, 0
         1, 0, 0, 0
         0, 0, 0, 1]
    y_ord = [1, 1, 0
             1, 0, 0
             0, 0, 0
             1, 1, 1]
    """
    # with tf.device('/cpu:0'):

    y_reg_shape = y_reg.get_shape().as_list()
    # if y_reg_shape[-1] == 1:
    y_reg = tf.one_hot(tf.cast(y_reg, tf.int32), axis=-1, depth=num_classes)
    # if len(y_reg_shape) > 2:
    N = reduce(lambda x, y: x * y, y_reg_shape[:-1])
    y_reg = tf.reshape(y_reg, [N, num_classes])

    y_reg_t = tf.transpose(y_reg, [1, 0])
    y_ord_t = 1.0 - tf.scan(lambda memo, x: tf.maximum(memo, x), y_reg_t)
    y_ord_t = tf.nn.embedding_lookup(y_ord_t, tf.range(num_classes - 1))  # slice first C-1 rows (cols in final matrix)
    y_ord = tf.transpose(y_ord_t, [1, 0])
    y_ord = tf.reshape(y_ord, y_reg_shape[:-1] + [num_classes-1])
    return y_ord

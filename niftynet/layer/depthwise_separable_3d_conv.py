# -*- coding: utf-8 -*-

"""
This is a modification of the SeparableConv3D code in Keras,
to perform just the Depthwise Convolution (1st step) of the
Depthwise Separable Convolution layer.
"""

from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from niftynet.layer.base_layer import TrainableLayer


def default_w_initializer():
    def _initializer(shape, dtype, partition_info):
        stddev = np.sqrt(2.0 / np.prod(shape[:-1]))
        from tensorflow.python.ops import random_ops
        return random_ops.truncated_normal(shape, 0.0, stddev, dtype=tf.float32)
        # return tf.truncated_normal_initializer(
        #    mean=0.0, stddev=stddev, dtype=tf.float32)

    return _initializer


def default_b_initializer():
    return tf.constant_initializer(0.0)


def preprocess_padding(padding):
    if padding.lower() in ('same', 'valid'):
        padding = padding.upper()
    else:
        raise ValueError('Invalid border mode:', padding)
    return padding


def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
    """Determines output length of a convolution given input length.

    # Arguments
        input_length: integer.
        filter_size: integer.
        padding: one of "same", "valid", "full".
        stride: integer.
        dilation: dilation rate, integer.

    # Returns
        The output length (integer).
    """
    if input_length is None:
        return None
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding == 'same':
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'causal':
        output_length = input_length
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    else:
        raise ValueError("Please ensure padding is one of 'same', 'valid', 'full', 'causal'")
    return (output_length + stride - 1) // stride


class DepthwiseConv3D(TrainableLayer):
    """Depthwise separable 3D convolution.
    Depthwise Separable convolutions consist in performing
    just the first step in a depthwise spatial convolution
    (which acts on each input channel separately).
    It does not perform the pointwise convolution (second step).
    The `depth_multiplier` argument controls how many
    output channels are generated per input channel in the depthwise step.
    # Arguments
        kernel_size: An integer or tuple/list of 3 integers, specifying the
            depth, width and height of the 3D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 3 integers,
            specifying the strides of the convolution along the depth, width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filterss_in * depth_multiplier`.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        depthwise_initializer: Initializer for the depthwise kernel matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        depthwise_regularizer: Regularizer function applied to
            the depthwise kernel matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        dialation_rate: List of ints.
                        Defines the dilation factor for each dimension in the
                        input. Defaults to (1,1,1)
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        depthwise_constraint: Constraint function applied to
            the depthwise kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        5D tensor with shape:
        `(batch, depth, channels, rows, cols)` if data_format='channels_first'
        or 5D tensor with shape:
        `(batch, depth, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        5D tensor with shape:
        `(batch, filters * depth, new_depth, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_depth, new_rows, new_cols, filters * depth)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='same',
                 depth_multiplier=1,
                 data_format='channels_last',
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 dilation_rate=(1, 1, 1),
                 depthwise_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 name='depthwise_separable_conv3d',
                 **kwargs):

        super(DepthwiseConv3D, self).__init__(name=name)
        if type(kernel_size) is int:
            kernel_size = [kernel_size] * 3

        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.depth_multiplier = depth_multiplier
        self.use_bias = use_bias
        self.initializers = {
            'w': w_initializer if w_initializer else default_w_initializer(),
            'b': b_initializer if b_initializer else default_b_initializer()}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        self.depthwise_initializer = tf.keras.initializers.get(depthwise_initializer)
        self.depthwise_regularizer = tf.keras.regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = tf.keras.constraints.get(depthwise_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.dilation_rate = dilation_rate
        self._padding = preprocess_padding(self.padding)
        self._strides = (1,) + self.strides + (1,)
        self._data_format = "NDHWC"  # todo -- why is this used twice?
        self._op = tf.make_template(name, self.layer_op, create_scope_now_=True)

    def layer_op(self, inputs, is_training):
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  self.kernel_size[2],
                                  1,
                                  self.depth_multiplier)

        depthwise_kernel = tf.get_variable(
            'depthwise_kernel',
            shape=depthwise_kernel_shape,
            initializer=self.initializers['w'],
            regularizer=self.regularizers['w'])

        if inputs.dtype == 'float64':
            inputs = tf.cast(inputs, 'float32')
        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, (0, 2, 3, 4, 1))

        if self.data_format == 'channels_last':
            b, d, h, w, c = inputs.shape
            channels_l = [tf.expand_dims(inputs[:, :, :, :, i], -1) for i in range(0, c)]
            dilation = (1,) + self.dilation_rate + (1,)
        else:
            b, c, d, h, w = inputs.shape
            channels_l = [tf.expand_dims(inputs[:, i, :, :, :], 1) for i in range(0, c)]
            dilation = self.dilation_rate + (1,) + (1,)

        outputs = tf.convert_to_tensor(
            [tf.nn.conv3d(inp_c, depthwise_kernel, strides=self._strides,
                          padding=self._padding, dilations=dilation,
                          data_format=self._data_format)
             for inp_c in channels_l])

        # Add original channels dim at the end of the tensor
        outputs = tf.transpose(outputs, (1, 2, 3, 4, 5, 0))
        # Convert to tensor [batch, depth, height, width, new_channels/depth_multiplier * original_channels]
        shape = outputs.get_shape().as_list()

        outputs = tf.reshape(outputs, [-1, shape[1], shape[2], shape[3], shape[4] * shape[5]])
        print('outputs.shape', outputs.shape)
        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def __call__(self, *args, **kwargs):
        return self._op(*args, **kwargs)

    def __str__(self):
        return self.to_string()

    def layer_scope(self):
        return self._op.variable_scope

    def to_string(self):
        layer_scope_name = self.layer_scope().name
        out_str = "\033[42m[Layer]\033[0m {}".format(layer_scope_name)
        if not self._op._variables_created:
            out_str += ' \033[46m(input undecided)\033[0m'
            return out_str
        return out_str

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            depth, rows, cols = input_shape[2:5]
            out_filters = input_shape[1] * self.depth_multiplier
        else:
            depth, rows, cols = input_shape[1:4]
            out_filters = input_shape[4] * self.depth_multiplier

        depth = conv_output_length(depth, self.kernel_size[0], self.padding, self.strides[0])
        rows = conv_output_length(rows, self.kernel_size[1], self.padding, self.strides[1])
        cols = conv_output_length(cols, self.kernel_size[2], self.padding, self.strides[2])

        if self.data_format == 'channels_first':
            return input_shape[0], out_filters, depth, rows, cols
        else:
            return input_shape[0], depth, rows, cols, out_filters

    def get_config(self):
        config = super(DepthwiseConv3D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = tf.keras.initializers.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = tf.keras.regularizers.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = tf.keras.constraints.serialize(self.depthwise_constraint)
        return config

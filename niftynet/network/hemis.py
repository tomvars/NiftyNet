# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from six.moves import range
import tensorflow as tf

from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.downsample import DownSampleLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.network.base_net import BaseNet
from niftynet.layer import layer_util
from niftynet.network.highres3dnet import HighResBlock
from niftynet.layer.dilatedcontext import DilatedTensor

class HeMIS(BaseNet):
    """
    Implementation of HeMIS: Hetero-Modal Image Segmentation:
        Havaei, M., Guizard, N., Chapados, N., & Bengio, Y. (2016, July 18).
         HeMIS: Hetero-Modal Image Segmentation. arXiv.org.
    Implementation uses the "Backend", "Abstraction", "Frontend" terminology of the
     paper to make it clearer.
    """
    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='HeMIS'):
        super(HeMIS, self).__init__(
            num_classes=num_classes,
            acti_func=acti_func,
            name=name,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer)

        self.backend_layers = [
            {
                'name': 'backend_conv_1',
                'n_features': 48,
                'kernel_size': 5,
                'acti_func': 'relu',
                'padding': 'ZERO'
            },
            {
                'name': 'backend_conv_2',
                'n_features': 48,
                'kernel_size': 5,
                'acti_func': 'relu',
                'padding': 'ZERO'
            },
            {
                'name': 'pooling',
                'n_features': 48,
                'kernel_size': 2,
                'stride': 1,
                'padding': 'SAME',
                'pooling_type': 'MAX'
            }
        ]
        self.frontend_layers = [
            {
                'name': 'frontend_conv_3',
                'n_features': 16,
                'kernel_size': 5,
                'acti_func': 'relu',
                'padding': 'ZERO'
            },
            {
                'name': 'frontend_conv_4',
                'n_features': num_classes,
                'kernel_size': 5,
                'acti_func': 'softmax',
                'padding': 'ZERO'
            }
        ]
        self.num_classes = num_classes

    def layer_op(self, input_tensor, is_training, layer_id=-1):
        n_modalities = input_tensor.shape.as_list()[-1]
        tf.logging.info('Input tensor dims: %s' % input_tensor.shape)
        # CAST TO 2D FOR SIMPLICITY
        # input_tensor = input_tensor[:, :, :, 50, :]

        backend_outputs = []
        # Loop through each modality, compute the backend tensor of each one.
        for modality in range(n_modalities):
            # _single_modality_backend_tensor = HeMISBackendBlock(layers=self.backend_layers,
            #                                                     name='HeMISBackendBlock_' + str(modality))
            _single_modality_backend_tensor = HighRes3dBackendBlock(name='HighRes3dBackendBlock_' + str(modality))
            tf.logging.info('modality tensor input: %s' % input_tensor[:, :, :, :, modality])
            # _tensor = _single_modality_backend_tensor(tf.expand_dims(input_tensor[:, :, :, :, modality], -1), is_training)
            _tensor = _single_modality_backend_tensor(input_tensor[:, :, :, :, modality], is_training)
            tf.logging.info('Modality tensor dims: %s' % _tensor.shape)
            backend_outputs.append(_tensor)

        full_backend_tensor = tf.concat([backend_outputs], axis=0)
        tf.logging.info('Full backend dims: %s' % full_backend_tensor.shape)
        exit()

        # full_backend_tensor = tf.transpose(full_backend_tensor, [1, 2, 3, 4, 5, 0])
        tf.logging.info('Full backend dims: %s' % full_backend_tensor.shape)
        abstraction_op = HeMISAbstractionBlock(pooling_type='average')
        abstraction_tensor = abstraction_op(full_backend_tensor, is_training)
        tf.logging.info('Abstraction output dims: %s' % abstraction_tensor.shape)
        # frontend_op = HeMISFrontendBlock(layers=self.frontend_layers)
        frontend_op = HighRes3dFrontendBlock(num_classes=self.num_classes)
        frontend_tensor = frontend_op(abstraction_tensor, is_training)
        tf.logging.info('Frontend output dims: %s' % frontend_tensor.shape)
        return frontend_tensor


class HeMISBackendBlock(TrainableLayer):
    """
    This class defines the entire HeMIS backend for one image modality.
    """
    def __init__(self,
                 layers,
                 w_initializer=None,
                 w_regularizer=None,
                 name='HeMISBackendBlock'):
        super(HeMISBackendBlock, self).__init__(name=name)
        self.layers = layers
        self.name = name
        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}

    def layer_op(self, input_tensor, is_training):
        """
        Function uses tf.cond to either compute or not compute representation C
        for a given input_tensor dimension.

        :param input_tensor: input_tensor is a single MR scan of  size HxWxD
        :return: CxN tensor where C is the number of features and N is the number of modalities.
        """
        params = self.layers[0]
        conv_layer_1 = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=params['acti_func'],
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        backend_output = conv_layer_1(input_tensor, is_training)
        params = self.layers[1]
        conv_layer_2 = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=params['acti_func'],
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        backend_output = conv_layer_2(backend_output, is_training)
        params = self.layers[2]
        pooling_layer = DownSampleLayer(
            func=params['pooling_type'],
            kernel_size=params['kernel_size'],
            stride=params['stride'],
            padding=params['padding'],
            name=params['name']
        )

        return tf.cond(tf.greater(tf.count_nonzero(input_tensor), 0),
                       true_fn=lambda: pooling_layer(backend_output),
                       false_fn=lambda: tf.zeros(backend_output.shape))

    def _print(self, list_of_layers):
        for (op, _) in list_of_layers:
            print(op)


class HighRes3dBackendBlock(BaseNet):
    def __init__(self,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='HighRes3DNet'):

        super(HighRes3dBackendBlock, self).__init__(
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.layers = [
            {'name': 'conv_0', 'n_features': 16, 'kernel_size': 3},
            {'name': 'res_1', 'n_features': 16, 'kernels': (3, 3), 'repeat': 3},
            {'name': 'res_2', 'n_features': 32, 'kernels': (3, 3), 'repeat': 3},
            {'name': 'res_3', 'n_features': 64, 'kernels': (3, 3), 'repeat': 3}]

    def layer_op(self, images, is_training, layer_id=-1):
        assert layer_util.check_spatial_dims(
            images, lambda x: x % 8 == 0)
        # go through self.layers, create an instance of each layer
        # and plugin data
        layer_instances = []

        ### first convolution layer
        params = self.layers[0]
        first_conv_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = first_conv_layer(images, is_training)
        layer_instances.append((first_conv_layer, flow))

        ### resblocks, all kernels dilated by 1 (normal convolution)
        params = self.layers[1]
        with DilatedTensor(flow, dilation_factor=1) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(
                    params['n_features'],
                    params['kernels'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        ### resblocks, all kernels dilated by 2
        params = self.layers[2]
        with DilatedTensor(flow, dilation_factor=2) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(
                    params['n_features'],
                    params['kernels'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        ### resblocks, all kernels dilated by 4
        params = self.layers[3]
        with DilatedTensor(flow, dilation_factor=4) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(
                    params['n_features'],
                    params['kernels'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))

        # set training properties
        if is_training:
            return layer_instances[-1][1]
        return layer_instances[layer_id][1]


class HeMISAbstractionBlock(TrainableLayer):
    def __init__(self,
                 pooling_type='average',
                 name='HeMISAbstractionBlock'):

        super(HeMISAbstractionBlock, self).__init__(name=name)

        self.pooling_type = pooling_type
        self.name = name

    def layer_op(self, input_tensor, is_training):
        """
        Function will drop all zero columns and compute E[C] and Var[C]
        :param backend_output: backend_output
        :return: 1xC tensor where C is the number of features.
        """
        # Omit zero columns from average
        # intermediate_tensor = tf.reduce_sum(tf.abs(input_tensor), 0)
        # zero_vector = tf.zeros(shape=(1, 1), dtype=tf.float32)
        # bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
        # omit_zero_columns = tf.boolean_mask(input_tensor, bool_mask)
        # Compute E[C]
        average_over_modalities, variance_between_modalities = tf.nn.moments(input_tensor, axes=[-1])
        abstraction_output = tf.concat([average_over_modalities, variance_between_modalities], axis=-1)
        return abstraction_output


class HeMISFrontendBlock(TrainableLayer):

    def __init__(self,
                 layers,
                 w_initializer=None,
                 w_regularizer=None,
                 name='HeMISFrontendBlock'):
        super(HeMISFrontendBlock, self).__init__(name=name)

        self.layers = layers
        self.name = name
        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}

    def layer_op(self, input_tensor, is_training):
        """
        Function will build the final layers of the network without the loss
        (Loss is added in the segmentation application)
        :param abstraction_output: abstraction_output
        :return: builds fronte
        """
        params = self.layers[0]
        conv_layer_1 = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=params['acti_func'],
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        frontend_output = conv_layer_1(input_tensor, is_training)
        params = self.layers[1]
        conv_layer_2 = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=params['acti_func'],
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        frontend_output = conv_layer_2(frontend_output, is_training)
        return frontend_output


class HighRes3dFrontendBlock(BaseNet):
    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='HighRes3DNet'):

        super(HighRes3dFrontendBlock, self).__init__(
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.layers = [
            {'name': 'conv_1', 'n_features': 80, 'kernel_size': 1},
            {'name': 'conv_2', 'n_features': num_classes, 'kernel_size': 1}]

    def layer_op(self, flow, is_training, layer_id=-1):
        # go through self.layers, create an instance of each layer
        # and plugin data
        layer_instances = []

        ### 1x1x1 convolution layer
        params = self.layers[0]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = fc_layer(flow, is_training)
        layer_instances.append((fc_layer, flow))

        ### 1x1x1 convolution layer
        params = self.layers[1]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=None,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = fc_layer(flow, is_training)
        layer_instances.append((fc_layer, flow))

        # set training properties
        if is_training:
            return layer_instances[-1][1]
        return layer_instances[layer_id][1]
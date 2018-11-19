# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from six.moves import range
import tensorflow as tf

from niftynet.network.base_net import BaseNet
from niftynet.layer import layer_util
from niftynet.contrib.pimms.resnet_plugin import ResNet
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.dilatedcontext import DilatedTensor
from niftynet.network.base_net import BaseNet
from niftynet.network.highres3dnet import HighResBlock
from niftynet.layer.base_layer import TrainableLayer


class SingletaskPIMMS3D(BaseNet):
    """
    Implementation of HeMIS: Hetero-Modal Image Segmentation:
        Havaei, M., Guizard, N., Chapados, N., & Bengio, Y. (2016, July 18).
         HeMIS: Hetero-Modal Image Segmentation. arXiv.org.
    Implementation uses the "Backend", "Abstraction", "Frontend" terminology of the
     paper to make it clearer.
    """

    def __init__(self,
                 num_classes,
                 n_modalities,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='HeMIS'):
        super(SingletaskPIMMS3D, self).__init__(
            num_classes=num_classes,
            acti_func=acti_func,
            name=name,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer)
        self.num_classes = num_classes
        self.n_modalities = n_modalities

    def layer_op(self, input_tensor, modality_slice, is_training, outputs_collector=None):
        n_modalities = self.n_modalities
        n_subj_in_batch = input_tensor.shape.as_list()[0]
        tf.logging.info('Input tensor dims: %s' % input_tensor.shape)
        tf.logging.info('modality_slice dims: %s' % modality_slice.shape)
        modality_slice_reshaped = tf.reshape(
            modality_slice, [modality_slice.shape[0], modality_slice.shape[1], modality_slice.shape[2], -1]
        )
        tf.logging.info('modality_slice_reshaped: %s' % modality_slice_reshaped.shape)
        n_ims_per_subj = modality_slice_reshaped.shape.as_list()[-1]
        with tf.variable_scope('modality_classifier') as scope:
            modality_classifier = ResNet(n_modalities,
                                         w_regularizer=self.regularizers['w'],
                                         b_regularizer=self.regularizers['b'],
                                         with_bn=True)
            modality_scores = modality_classifier(tf.einsum('ijkl->ljki', modality_slice_reshaped), is_training)
            modality_scores = tf.Print(modality_scores, [modality_scores])
        tf.logging.info('modality_scores: %s' % modality_scores.shape)
        modality_scores_modified = tf.reshape(modality_scores, [n_subj_in_batch, modality_slice.shape[3], modality_slice.shape[-1], -1])
        modality_scores_modified = tf.Print(modality_scores_modified, [tf.argmax(modality_scores_modified[0, ...], axis=-1)[:, 0]])
        tf.logging.info('modality_scores_modified: %s' % modality_scores_modified.shape)
        modality_scores_modified_after_mean = tf.reduce_mean(modality_scores_modified, axis=1)
        tf.logging.info('modality_scores_modified_after_mean: %s' % modality_scores_modified_after_mean.shape)
        modality_tensor = tf.expand_dims(tf.expand_dims(tf.expand_dims(modality_scores_modified_after_mean, axis=2), axis=2),
                                         axis=2)
        print('modality_tensor', modality_tensor.shape)
        expanded_input_tensor = tf.expand_dims(input_tensor, axis=1)
        print('expanded_input_tensor', expanded_input_tensor.shape)
        attention_tensor = tf.reduce_sum(tf.multiply(expanded_input_tensor, modality_tensor), axis=-1)
        print('attention_tensor', attention_tensor.shape)
        normalization_tensor = tf.reduce_sum(modality_tensor, axis=-1)
        print('normalization_tensor', normalization_tensor.shape)
        attention_tensor = attention_tensor / normalization_tensor
        attention_tensor = tf.transpose(attention_tensor, [0, 2, 3, 4, 1], name='attention_tensor')
        print('attention_tensor', attention_tensor.shape)
        backend_outputs = []
        # Loop through each modality, compute the backend tensor of each one.
        for modality in range(n_modalities):
            _single_modality_backend_tensor = HighRes3DNetSmallBackendBlock(name='HeMISBackendBlock_' + str(modality),
                                                                            w_regularizer=self.regularizers['w'])
            tf.logging.info('attention_tensor input: %s' % attention_tensor[..., modality])
            _tensor = _single_modality_backend_tensor(tf.expand_dims(attention_tensor[..., modality], -1), is_training)
            # _tensor = _single_modality_backend_tensor(input_tensor[:, :, :, :, modality], is_training)
            tf.logging.info('Modality tensor dims: %s' % _tensor.shape)
            backend_outputs.append(_tensor)

        full_backend_tensor = tf.concat([backend_outputs], axis=0)
        tf.logging.info('Full backend dims: %s' % full_backend_tensor.shape)
        new_shape = list(range(len(full_backend_tensor.shape)))
        new_shape = new_shape[1:] + new_shape[:1]
        full_backend_tensor = tf.transpose(full_backend_tensor, new_shape)
        tf.logging.info('Full backend dims: %s' % full_backend_tensor.shape)
        abstraction_op = HeMISAbstractionBlock(pooling_type='average')
        abstraction_tensor = abstraction_op(full_backend_tensor, is_training)
        tf.logging.info('Abstraction output dims: %s' % abstraction_tensor.shape)
        segmentation_op = HighRes3dFrontendBlock(num_classes=self.num_classes,
                                                 w_regularizer=self.regularizers['w'])
        segmentation_tensor = segmentation_op(abstraction_tensor, is_training)
        tf.logging.info('Segmentation frontend output dims: %s' % segmentation_tensor.shape)
        classification_tensor = tf.expand_dims(modality_scores, axis=0)
        tf.logging.info('Classification tensor output dims: %s' % classification_tensor.shape)
        return segmentation_tensor, classification_tensor


class HighRes3DNetSmallBackendBlock(BaseNet):
    """
    implementation of HighRes3DNet:

        Li et al., "On the compactness, efficiency, and representation of 3D
        convolutional networks: Brain parcellation as a pretext task", IPMI '17

    (This is smaller model with an initial stride-2 convolution)
    """

    def __init__(self,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='HighRes3DNetSmall'):

        super(HighRes3DNetSmallBackendBlock, self).__init__(
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
            {'name': 'res_3', 'n_features': 64, 'kernels': (3, 3), 'repeat': 3},
            {'name': 'conv_1', 'n_features': 80, 'kernel_size': 3}]

        # self.layers = [
        #     {'name': 'conv_0', 'n_features': 16, 'kernel_size': 3},
        #     {'name': 'res_1', 'n_features': 16, 'kernels': (3, 3), 'repeat': 3},
        #     {'name': 'res_2', 'n_features': 32, 'kernels': (3, 3), 'repeat': 3},
        #     {'name': 'res_3', 'n_features': 64, 'kernels': (3, 3), 'repeat': 3},
        #     {'name': 'conv_1', 'n_features': 64, 'kernel_size': 3},
        #     {'name': 'conv_2', 'n_features': 64, 'kernel_size': 1},
        #     {'name': 'conv_3', 'n_features': num_classes, 'kernel_size': 1}]

    def layer_op(self, images, is_training=True, layer_id=-1, **unused_kwargs):
        assert (layer_util.check_spatial_dims(
            images, lambda x: x % 8 == 0))
        # go through self.layers, create an instance of each layer
        # and plugin data
        layer_instances = []

        ### first convolution layer
        params = self.layers[0]
        first_conv_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            stride=2,
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
        flow = dilated.tensor

        ### 1x1x1 convolution layer
        params = self.layers[4]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = fc_layer(flow, is_training)
        layer_instances.append((fc_layer, flow))

        ### 3x3x3 deconvolution layer
        params = self.layers[4]
        fc_layer = DeconvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=3,
            stride=2,
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='deconv')
        flow = fc_layer(flow, is_training)
        layer_instances.append((fc_layer, flow))

        return tf.cond(tf.greater(tf.count_nonzero(images), 0),
                       true_fn=lambda: flow,
                       false_fn=lambda: tf.zeros(flow.shape))


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

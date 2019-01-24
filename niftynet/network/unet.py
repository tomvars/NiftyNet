# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.downsample import DownSampleLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.layer.crop import CropLayer
from niftynet.layer.linear_resize import LinearResizeLayer
from niftynet.utilities.util_common import look_up_operations


class UNet3D(TrainableLayer):
    """
    reimplementation of 3D U-net
      Çiçek et al., "3D U-Net: Learning dense Volumetric segmentation from
      sparse annotation", MICCAI '16
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='leakyrelu',
                 name='UNet'):
        super(UNet3D, self).__init__(name=name)

        self.n_features = [30, 60, 120, 240, 480]
        self.acti_func = acti_func
        self.num_classes = num_classes

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

        print('using {}'.format(name))

    def layer_op(self, images, is_training=True, layer_id=-1, **unused_kwargs):
        # image_size  should be divisible by 8
        assert layer_util.check_spatial_dims(images, lambda x: (x - 4) % 8 == 0)
        # assert layer_util.check_spatial_dims(images, lambda x: x >= 89)
        block_layer = UNetBlock('DOWNSAMPLE',
                                (self.n_features[0], self.n_features[1]),
                                (3, 3), with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='L1')
        pool_1, conv_1 = block_layer(images, is_training)
        print(block_layer)

        block_layer = UNetBlock('DOWNSAMPLE',
                                (self.n_features[1], self.n_features[2]),
                                (3, 3), with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='L2')
        pool_2, conv_2 = block_layer(pool_1, is_training)
        print(block_layer)

        block_layer = UNetBlock('DOWNSAMPLE',
                                (self.n_features[2], self.n_features[3]),
                                (3, 3), with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='L3')
        pool_3, conv_3 = block_layer(pool_2, is_training)
        print(block_layer)

        block_layer = UNetBlock('UPSAMPLE',
                                (self.n_features[3], self.n_features[4]),
                                (3, 3), with_downsample_branch=False,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='L4')
        up_3, _ = block_layer(pool_3, is_training)
        print(block_layer)

        block_layer = UNetBlock('UPSAMPLE',
                                (self.n_features[3], self.n_features[3]),
                                (3, 3), with_downsample_branch=False,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='R3')

        conv_3 = CropLayer(4)(conv_3)
        concat_3 = ElementwiseLayer('CONCAT')(conv_3, up_3)
        up_2, _ = block_layer(concat_3, is_training)
        print(block_layer)

        block_layer = UNetBlock('UPSAMPLE',
                                (self.n_features[2], self.n_features[2]),
                                (3, 3), with_downsample_branch=False,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='R2')
        conv_2 = CropLayer(16)(conv_2)
        concat_2 = ElementwiseLayer('CONCAT')(conv_2, up_2)
        up_1, _ = block_layer(concat_2, is_training)
        print(block_layer)

        block_layer = UNetBlock('NONE',
                                (self.n_features[1],
                                 self.n_features[1],
                                 self.num_classes),
                                (3, 3, 1),
                                with_downsample_branch=True,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='R1_FC')

        conv_1 = CropLayer(40)(conv_1)
        concat_1 = ElementwiseLayer('CONCAT')(conv_1, up_1)

        # for the last layer, upsampling path is not used
        _, output_tensor = block_layer(concat_1, is_training)
        print(block_layer)
        return output_tensor


SUPPORTED_OP = set(['DOWNSAMPLE', 'UPSAMPLE', 'NONE'])


class UNetBlock(TrainableLayer):
    def __init__(self,
                 func,
                 n_chns,
                 kernels,
                 w_initializer=None,
                 w_regularizer=None,
                 with_downsample_branch=False,
                 acti_func='relu',
                 name='UNet_block'):

        super(UNetBlock, self).__init__(name=name)

        self.func = look_up_operations(func.upper(), SUPPORTED_OP)

        self.kernels = kernels
        self.n_chns = n_chns
        self.with_downsample_branch = with_downsample_branch
        self.acti_func = acti_func

        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}

    def layer_op(self, input_tensor, is_training):
        output_tensor = input_tensor
        for (kernel_size, n_features) in zip(self.kernels, self.n_chns):
            conv_op = ConvolutionalLayer(n_output_chns=n_features,
                                         kernel_size=kernel_size,
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         acti_func=self.acti_func,
                                         name='{}'.format(n_features),
                                         featnorm_type='batch',
                                         moving_decay=0.99,
                                         padding="VALID")
            output_tensor = conv_op(output_tensor, is_training)

        if self.with_downsample_branch:
            branch_output = output_tensor
        else:
            branch_output = None

        if self.func == 'DOWNSAMPLE':
            downsample_op = DownSampleLayer('MAX',
                                            kernel_size=2,
                                            stride=2,
                                            name='down_2x2')
            output_tensor = downsample_op(output_tensor)

        elif self.func == 'UPSAMPLE':
            up_shape = [2 * int(output_tensor.shape[i]) for i in (1, 2, 3)]
            upsample_op = LinearResizeLayer(up_shape)
            output_tensor = upsample_op(output_tensor)
        elif self.func == 'NONE':
            pass  # do nothing
        return output_tensor, branch_output


if __name__ == "__main__":
    import os
    import tensorflow as tf

    import sys

    sys.path.append('../../')
    import numpy as np

    img = np.random.random([1, 132, 132, 116, 2])
    tf_img = tf.constant(img, dtype=tf.float32)

    sess = tf.InteractiveSession()
    unet = UNet3D(img.shape[-1])
    unet.layer_op(tf_img)
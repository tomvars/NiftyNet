# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.depthwise_separable_3d_conv import DepthwiseConv3D
from niftynet.network.base_net import BaseNet


class ToyNet(BaseNet):
    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='ToyNet'):

        super(ToyNet, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.hidden_features = 10

    def layer_op(self, images, is_training=True, **unused_kwargs):
        print('images.shape', images.shape)
        conv_1 = DepthwiseConv3D(kernel_size=(3, 3, 3),
                                 w_initializer=self.initializers['w'],
                                 w_regularizer=self.regularizers['w'],
                                 b_initializer=self.initializers['b'],
                                 b_regularizer=self.regularizers['b'],
                                 acti_func='relu',
                                 depth_multiplier=32,
                                 data_format='channels_last',
                                 name='conv_input')

        conv_2 = DepthwiseConv3D(kernel_size=(1, 1, 1),
                                 w_initializer=self.initializers['w'],
                                 w_regularizer=self.regularizers['w'],
                                 b_initializer=self.initializers['b'],
                                 b_regularizer=self.regularizers['b'],
                                 acti_func=None,
                                 depth_multiplier=2,
                                 data_format='channels_last',
                                 name='conv_output')
        conv_3 = ConvolutionalLayer(self.num_classes,
                                    kernel_size=1,
                                    w_initializer=self.initializers['w'],
                                    w_regularizer=self.regularizers['w'],
                                    b_initializer=self.initializers['b'],
                                    b_regularizer=self.regularizers['b'],
                                    acti_func=None,
                                    name='conv_output')


        flow = conv_1(images, is_training)
        print('flow1', flow)
        flow = conv_2(flow, is_training)
        print('flow2', flow)
        flow = conv_3(flow, is_training)
        print('flow3', flow)
        return flow

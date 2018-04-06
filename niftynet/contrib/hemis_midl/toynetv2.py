# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
from niftynet.contrib.hemis_midl.convolution import ConvLayer
from niftynet.layer.fully_connected import FCLayer
from niftynet.network.base_net import BaseNet
from niftynet.layer.activation import ActiLayer
from niftynet.layer.downsample import DownSampleLayer


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

    def layer_op(self, images, is_training):

        conv_1a = ConvLayer(64,
                           kernel_size=3,
                           w_initializer=self.initializers['w'],
                           w_regularizer=self.regularizers['w'],
                           b_initializer=self.initializers['b'],
                           b_regularizer=self.regularizers['b'],
                           name='conv_output')
        conv_1b = ConvLayer(64,
                           kernel_size=3,
                           w_initializer=self.initializers['w'],
                           w_regularizer=self.regularizers['w'],
                           b_initializer=self.initializers['b'],
                           b_regularizer=self.regularizers['b'],
                           name='conv_output')
        conv_2a = ConvLayer(128,
                           kernel_size=3,
                           w_initializer=self.initializers['w'],
                           w_regularizer=self.regularizers['w'],
                           b_initializer=self.initializers['b'],
                           b_regularizer=self.regularizers['b'],
                           name='conv_output')
        conv_2b = ConvLayer(128,
                           kernel_size=3,
                           w_initializer=self.initializers['w'],
                           w_regularizer=self.regularizers['w'],
                           b_initializer=self.initializers['b'],
                           b_regularizer=self.regularizers['b'],
                           name='conv_output')
        conv_3a = ConvLayer(256,
                           kernel_size=3,
                           w_initializer=self.initializers['w'],
                           w_regularizer=self.regularizers['w'],
                           b_initializer=self.initializers['b'],
                           b_regularizer=self.regularizers['b'],
                           name='conv_output')
        conv_3b = ConvLayer(256,
                           kernel_size=3,
                           w_initializer=self.initializers['w'],
                           w_regularizer=self.regularizers['w'],
                           b_initializer=self.initializers['b'],
                           b_regularizer=self.regularizers['b'],
                           name='conv_output')
        conv_4a = ConvLayer(512,
                           kernel_size=3,
                           w_initializer=self.initializers['w'],
                           w_regularizer=self.regularizers['w'],
                           b_initializer=self.initializers['b'],
                           b_regularizer=self.regularizers['b'],
                           name='conv_output')
        conv_4b = ConvLayer(512,
                           kernel_size=3,
                           w_initializer=self.initializers['w'],
                           w_regularizer=self.regularizers['w'],
                           b_initializer=self.initializers['b'],
                           b_regularizer=self.regularizers['b'],
                           name='conv_output')
        pooling_layer = DownSampleLayer('MAX',
                                        kernel_size=2,
                                        stride=2)

        flow = conv_1a(images)
        flow = ActiLayer('relu')(flow)
        flow = conv_1b(flow)
        flow = ActiLayer('relu')(flow)
        flow = pooling_layer(flow)

        flow = conv_2a(flow)
        flow = ActiLayer('relu')(flow)
        flow = conv_2b(flow)
        flow = ActiLayer('relu')(flow)
        flow = pooling_layer(flow)

        flow = conv_3a(flow)
        flow = ActiLayer('relu')(flow)
        flow = conv_3b(flow)
        flow = ActiLayer('relu')(flow)
        flow = pooling_layer(flow)

        flow = conv_4a(flow)
        flow = ActiLayer('relu')(flow)
        flow = conv_4b(flow)
        flow = pooling_layer(flow)

        flow = FCLayer(4096)(flow)
        flow = ActiLayer('relu')(flow)
        flow = FCLayer(4096)(flow)
        flow = ActiLayer('relu')(flow)
        flow = FCLayer(self.num_classes)(flow)
        flow = ActiLayer('softmax')(flow)
        return flow

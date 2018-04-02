# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
from niftynet.contrib.hemis_midl.convolution import ConvLayer
from niftynet.layer.fully_connected import FCLayer
from niftynet.network.base_net import BaseNet
from niftynet.layer.activation import ActiLayer


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
        conv_1 = ConvLayer(self.hidden_features,
                                    kernel_size=3,
                                    w_initializer=self.initializers['w'],
                                    w_regularizer=self.regularizers['w'],
                                    b_initializer=self.initializers['b'],
                                    b_regularizer=self.regularizers['b'],
                                    name='conv_input')

        conv_2 = ConvLayer(self.num_classes,
                                    kernel_size=3,
                                    w_initializer=self.initializers['w'],
                                    w_regularizer=self.regularizers['w'],
                                    b_initializer=self.initializers['b'],
                                    b_regularizer=self.regularizers['b'],
                                    name='conv_output')

        flow = conv_1(images)
        flow = ActiLayer('relu')(flow)
        flow = conv_2(flow)
        flow = FCLayer(self.num_classes)(flow)
        flow = ActiLayer('softmax')(flow)
        return flow

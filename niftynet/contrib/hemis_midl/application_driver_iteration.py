# -*- coding: utf-8 -*-
"""
This module defines a general procedure for running applications.

Example usage::
    app_driver = ApplicationDriver()
    app_driver.initialise_application(system_param, input_data_param)
    app_driver.run_application()

``system_param`` and ``input_data_param`` should be generated using:
``niftynet.utilities.user_parameters_parser.run()``
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import itertools

import tensorflow as tf
from blinker import signal
import numpy as np


from niftynet.engine.application_factory import ApplicationFactory
from niftynet.engine.application_iteration import IterationMessage
from niftynet.engine.application_variables import \
    CONSOLE, NETWORK_OUTPUT, TF_SUMMARIES
from niftynet.engine.application_variables import \
    GradientsCollector, OutputsCollector, global_vars_init_or_restore
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.io.image_sets_partitioner import TRAIN, VALID, INFER
from niftynet.io.misc_io import get_latest_subfolder, touch_folder
from niftynet.layer.bn import BN_COLLECTION
from niftynet.utilities.util_common import set_cuda_device, traverse_nested
from niftynet.engine.application_driver import ApplicationDriver

FILE_PREFIX = 'model.ckpt'

# pylint: disable=too-many-instance-attributes
class ApplicationDriverStep(ApplicationDriver):
    """
    This class initialises an application by building a TF graph,
    and maintaining a session and coordinator. It controls the
    starting/stopping of an application. Applications should be
    implemented by inheriting ``niftynet.application.base_application``
    to be compatible with this driver.
    """

    # pylint: disable=too-many-instance-attributes

    pre_train_iter = signal('pre_train_iter')
    post_train_iter = signal('post_train_iter')
    pre_validation_iter = signal('pre_validation_iter')
    post_validation_iter = signal('post_validation_iter')
    pre_infer_iter = signal('pre_infer_iter')
    post_infer_iter = signal('post_infer_iter')
    post_training = signal('post_training')

    def __init__(self):
        ApplicationDriver.__init__(self)

    def _loop(self, iteration_generator, sess, loop_status):
        for iter_msg in iteration_generator:
            if self._coord.should_stop():
                break
            if iter_msg.should_stop:
                break
            loop_status['current_iter'] = iter_msg.current_iter
            iter_msg.pre_iter.send(iter_msg)

            iter_msg.ops_to_run[NETWORK_OUTPUT] = \
                self.outputs_collector.variables(NETWORK_OUTPUT)
            # if self.is_training:
            #     iter_msg.data_feed_dict[iter_msg.ops_to_run['niftynetout']['iter']] = [[iter_msg.current_iter]]
            graph_output = sess.run(iter_msg.ops_to_run,
                                    feed_dict=iter_msg.data_feed_dict)
            iter_msg.current_iter_output = graph_output
            iter_msg.status = self.app.interpret_output(
                iter_msg.current_iter_output[NETWORK_OUTPUT]
            )

            iter_msg.post_iter.send(iter_msg)

            if iter_msg.should_stop:
                break
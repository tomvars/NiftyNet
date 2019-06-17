# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
import numpy as np
import uuid
import os

from tests.application_driver_test import get_initialised_driver, SYSTEM_PARAM
from niftynet.engine.application_iteration import IterationMessage
from niftynet.engine.signal import SESS_STARTED, ITER_FINISHED, VALID
from niftynet.utilities.util_common import ParserNamespace

class WholeVolumeTest(tf.test.TestCase):
    def test_init(self):
        ITER_FINISHED.connect(self.iteration_listener)
        model_dir = os.path.join('.', 'testing_data', 'tmp', str(uuid.uuid4()))
        os.makedirs(model_dir)
        SYSTEM_PARAM['SYSTEM'].model_dir = model_dir
        SYSTEM_PARAM['TRAINING'].do_whole_volume_validation = True
        SYSTEM_PARAM['TRAINING'].validation_every_n = 5
        SYSTEM_PARAM['TRAINING'].validation_max_iter = 10
        SYSTEM_PARAM['TRAINING'].max_iter = 100
        SYSTEM_PARAM['INFERENCE'] = ParserNamespace(
            border=(0, 0, 0),
            save_seg_dir='./output/unet',
            output_interp_order=0,
            spatial_window_size=(32, 32, 32)
        )
        app_driver = get_initialised_driver(system_param=SYSTEM_PARAM)
        app_driver.load_event_handlers(
            ['niftynet.engine.handler_model.ModelRestorer',
             'niftynet.engine.handler_console.ConsoleLogger',
             'niftynet.engine.handler_sampler.SamplerThreading'])
        graph = app_driver.create_graph(app_driver.app, 1, True)
        with self.test_session(graph=graph) as sess:
            iteration_messages = app_driver._generator(**vars(app_driver))()
            for msg in iteration_messages:
                SESS_STARTED.send(app_driver.app, iter_msg=None)
                app_driver.loop(app_driver.app, [msg])
                print(msg)
        app_driver.app.stop()
        ITER_FINISHED.disconnect(self.iteration_listener)

    def iteration_listener(self, sender, **msg):
        msg = msg['iter_msg']
        self.assertRegexpMatches(msg.to_console_string(), '.*total_loss.*')
        if msg.current_iter > 1:
            self.assertTrue(isinstance(sender.performance_history, list))
            self.assertTrue(len(sender.performance_history) <= sender.patience)
            self.assertTrue(all([isinstance(p, np.float32) for p in sender.performance_history]))


if __name__ == "__main__":
    tf.test.main()

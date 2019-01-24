# -*- coding: utf-8 -*-
"""
This module implements a network output interpreter.
"""
import numpy as np

from niftynet.engine.application_variables import NETWORK_OUTPUT
from niftynet.engine.signal import ITER_STARTED, ITER_FINISHED


class ApplyDistillationLoss(object):
    """
    This class handles iteration events to interpret output.
    """

    def __init__(self, **_unused):
        self.last_output = None
        ITER_STARTED.connect(self.set_tensor_values)
        ITER_FINISHED.connect(self.gather_outputs)

    def set_tensor_values(self, sender, **msg):
        """
        Event handler to add all tensors to evaluate to the iteration message.
        The driver will fetch tensors' values from
        ``_iter_msg.ops_to_run``.

        :param sender: a niftynet.application instance
        :param msg: an iteration message instance
        :return:
        """
        _iter_msg = msg['iter_msg']
        ops_to_run = sender.outputs_collector.variables(NETWORK_OUTPUT)
        if _iter_msg.current_iter > 100 and self.last_output is not None:
            _iter_msg.data_feed_dict[ops_to_run['last_brain_parcellation_activation']] =\
                self.last_output['current_brain_parcellation_activation']
            _iter_msg.data_feed_dict[ops_to_run['last_lesion_segmentation_activation']] = \
                self.last_output['current_lesion_segmentation_activation']
            _iter_msg.data_feed_dict[ops_to_run['last_tumour_segmentation_activation']] = \
                self.last_output['current_tumour_segmentation_activation']
        else:
            _iter_msg.data_feed_dict[ops_to_run['last_brain_parcellation_activation']] =\
                np.zeros(shape=ops_to_run['last_brain_parcellation_activation'].shape.as_list(),
                         dtype=np.float)
            _iter_msg.data_feed_dict[ops_to_run['last_lesion_segmentation_activation']] = \
                np.zeros(shape=ops_to_run['last_lesion_segmentation_activation'].shape.as_list(),
                         dtype=np.float)
            _iter_msg.data_feed_dict[ops_to_run['last_tumour_segmentation_activation']] = \
                np.zeros(shape=ops_to_run['last_tumour_segmentation_activation'].shape.as_list(),
                         dtype=np.float)

    def gather_outputs(self, _sender, **msg):
        """
        Calling sender application to interpret evaluated tensors.
        Set ``_iter_msg.should_stop`` to a True value
        if it's an end of the engine loop.

        See also:
        ``niftynet.engine.application_driver.loop``

        :param sender: a niftynet.application instance
        :param msg: an iteration message instance
        :return:
        """
        _iter_msg = msg['iter_msg']
        self.last_output[_iter_msg.current_iter] = _iter_msg.current_iter_output[NETWORK_OUTPUT]

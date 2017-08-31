# -*- coding: utf-8 -*-
"""
windows aggregator decode sampling grid coordinates and image id from
batch data, forms image level output and write to hard drive
"""
from __future__ import absolute_import, print_function, division

import os

import numpy as np

import niftynet.io.misc_io as misc_io
from niftynet.engine.windows_aggregator_base import ImageWindowsAggregator
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer
from niftynet.layer.pad import PadLayer


class GridSamplesAggregator(ImageWindowsAggregator):
    def __init__(self,
                 image_reader,
                 name='image',
                 output_path='./',
                 window_border=(),
                 interp_order=0):
        ImageWindowsAggregator.__init__(self, image_reader=image_reader)
        self.name = name
        self.image_out = None
        self.output_path = os.path.abspath(output_path)
        self.window_border = window_border
        self.output_interp_order = interp_order

    def decode_batch(self, window, location):
        n_samples = location.shape[0]
        window, location = self.crop_batch(window, location, self.window_border)

        for batch_id in range(n_samples):
            image_id, x_, y_, z_, _x, _y, _z = location[batch_id, :]
            if image_id != self.image_id:
                # image name changed:
                #    save current image and create an empty image
                self._save_current_image()
                if self._is_stopping_signal(location[batch_id]):
                    return False
                self.image_out = self._initialise_empty_image(
                    image_id=image_id,
                    n_channels=window.shape[-1],
                    dtype=window.dtype)
            self.image_out[x_:_x, y_:_y, z_:_z, ...] = window[batch_id, ...]
        return True

    def _initialise_empty_image(self, image_id, n_channels, dtype=np.float):
        self.image_id = image_id
        spatial_shape = self.input_image[self.name].shape[:3]
        output_image_shape = spatial_shape + (n_channels,)
        empty_image = np.zeros(output_image_shape, dtype=dtype)

        for layer in self.reader.preprocessors:
            if isinstance(layer, PadLayer):
                empty_image, _ = layer(empty_image)
        return empty_image

    def _save_current_image(self):
        if self.input_image is None:
            return

        for layer in reversed(self.reader.preprocessors):
            if isinstance(layer, PadLayer):
                self.image_out, _ = layer.inverse_op(self.image_out)
            if isinstance(layer, DiscreteLabelNormalisationLayer):
                self.image_out, _ = layer.inverse_op(self.image_out)
        subject_name = self.reader.get_subject_id(self.image_id)
        filename = "{}_niftynet_out.nii.gz".format(subject_name)
        source_image_obj = self.input_image['image']
        misc_io.save_data_array(self.output_path,
                                filename,
                                self.image_out,
                                source_image_obj,
                                self.output_interp_order)
        return


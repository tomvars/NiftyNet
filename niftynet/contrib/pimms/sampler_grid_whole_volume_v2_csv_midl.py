# -*- coding: utf-8 -*-
"""
Sampling image by a sliding window.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.ndimage
import tensorflow as tf
from tensorflow.python.data.util import nest

from niftynet.engine.image_window_dataset import ImageWindowDataset
from niftynet.contrib.csv_reader.sampler_csv_rows import ImageWindowDatasetCSV
from niftynet.engine.image_window import N_SPATIAL, LOCATION_FORMAT


# pylint: disable=too-many-locals
class GridSampler(ImageWindowDatasetCSV):
    """
    This class generators ND image samples with a sliding window.
    """

    def __init__(self,
                 reader,
                 csv_reader,
                 window_sizes,
                 idxs_to_drop=None,
                 permuted_indices=None,
                 batch_size=1,
                 spatial_window_size=None,
                 window_border=None,
                 queue_length=1,
                 smaller_final_batch_mode='dynamic',
                 name='grid_sampler'):

        # override all spatial window defined in input
        # modalities sections
        # this is useful when do inference with a spatial window
        # which is different from the training specifications
        ImageWindowDatasetCSV.__init__(
            self,
            csv_reader=csv_reader,
            reader=reader,
            window_sizes=spatial_window_size or window_sizes,
            batch_size=batch_size,
            windows_per_image=1,
            queue_length=queue_length,
            shuffle=False,
            epoch=1,
            smaller_final_batch_mode=smaller_final_batch_mode,
            name=name)
        self.idxs_to_drop = idxs_to_drop
        self.permuted_indices = permuted_indices
        self.border_size = window_border or (0, 0, 0)
        assert isinstance(self.border_size, (list, tuple)), \
            "window_border should be a list or tuple"
        while len(self.border_size) < N_SPATIAL:
            self.border_size = tuple(self.border_size) + \
                               (self.border_size[-1],)
        self.border_size = self.border_size[:N_SPATIAL]
        self.no_more_samples = False
        tf.logging.info('initialised window instance')
        tf.logging.info("initialised grid sampler %s", self.window.shapes)

    def layer_op(self):
        while True:
            image_id, data, interp_orders = self.reader(idx=None, shuffle=False)
            if not data:
                self.reader.reset()
                self.no_more_samples = True
                break
            tf.logging.info('Called Sampler and self.idxs_to_drop is set to {}'.format(self.idxs_to_drop))
            ##### Deterministic drop of modalities according to params#####
            if self.idxs_to_drop:
                assert isinstance(self.idxs_to_drop, tuple)
                num_modalities = data['image'].shape[-1]
                data_shape_without_modality = list(data['image'].shape)[:-1]
                dropped_indices = []
                for idx_to_drop in self.idxs_to_drop:
                    print('DROPPING these modalities {}'.format(idx_to_drop))
                    data['image'][..., idx_to_drop] = np.zeros(shape=data_shape_without_modality)
                    dropped_indices.append(idx_to_drop)
                # Randomly permute the inputs
                permuted_indices = np.random.permutation(range(num_modalities))
                data['image'] = data['image'][..., permuted_indices]
            ########################################################
            image_shapes = {name: data[name].shape
                            for name in self.window.names}
            static_window_shapes = self.window.match_image_shapes(image_shapes)
            coordinates = grid_spatial_coordinates(
                image_id, image_shapes, static_window_shapes, self.border_size)

            # extend the number of sampling locations to be divisible
            # by batch size
            n_locations = list(coordinates.values())[0].shape[0]
            extra_locations = 0
            if (n_locations % self.batch_size) > 0:
                extra_locations = \
                    self.batch_size - n_locations % self.batch_size
            total_locations = n_locations + extra_locations

            tf.logging.info(
                'grid sampling image sizes: %s', image_shapes)
            tf.logging.info(
                'grid sampling window sizes: %s', static_window_shapes)
            if extra_locations > 0:
                tf.logging.info(
                    "yielding %s locations from image, "
                    "extended to %s to be divisible by batch size %s",
                    n_locations, total_locations, self.batch_size)
            else:
                tf.logging.info(
                    "yielding %s locations from image", n_locations)
            for i in range(total_locations):
                idx = i % n_locations
                print('Yielding image_id={} at {} / {}'.format(image_id, i+1, total_locations))
                #  initialise output dict
                output_dict = {}
                for name in list(data):
                    assert coordinates[name].shape[0] == n_locations, \
                        "different number of grid samples from the input" \
                        "images, don't know how to combine them in the queue"
                    x_start, y_start, z_start, x_end, y_end, z_end = \
                        coordinates[name][idx, 1:]
                    try:
                        image_window = data[name][
                            x_start:x_end, y_start:y_end, z_start:z_end, ...]
                    except ValueError:
                        tf.logging.fatal(
                            "dimensionality miss match in input volumes, "
                            "please specify spatial_window_size with a "
                            "3D tuple and make sure each element is "
                            "smaller than the image length in each dim.")
                        raise
                    # fill output dict with data
                    coord_key = LOCATION_FORMAT.format(name)
                    image_key = name
                    output_dict[coord_key] = coordinates[name][idx:idx+1, ...]
                    output_dict[image_key] = image_window[np.newaxis, ...]
                    if self.csv_reader is not None:
                        _, label_dict, _ = self.csv_reader(idx=image_id)
                        output_dict.update(label_dict)
                        name = 'modality_label'
                        # for name in self.csv_reader.task_param.keys():
                        output_dict[name + '_location'] = output_dict['image_location']
                yield output_dict

    @property
    def tf_shapes(self):
        """
        returns a dictionary of sampler output tensor shapes
        """
        assert self.window, 'Unknown output shapes: self.window not initialised'
        shape_dict = self.window.tf_shapes
        if self.csv_reader is not None:
            shape_dict.update(self.csv_reader.tf_shapes)
        return shape_dict

    @property
    def tf_dtypes(self):
        """
        returns a dictionary of sampler output tensorflow dtypes
        """
        assert self.window, 'Unknown output shapes: self.window not initialised'
        shape_dict = self.window.tf_dtypes
        if self.csv_reader is not None:
            shape_dict.update(self.csv_reader.tf_dtypes)
        return shape_dict

def grid_spatial_coordinates(subject_id, img_sizes, win_sizes, border_size):
    """
    This function generates all coordinates of feasible windows, with
    step sizes specified in grid_size parameter.

    The border size changes the sampling locations but not the
    corresponding window sizes of the coordinates.

    :param subject_id: integer value indicates the position of of this
        image in ``image_reader.file_list``
    :param img_sizes: a dictionary of image shapes, ``{input_name: shape}``
    :param win_sizes: a dictionary of window shapes, ``{input_name: shape}``
    :param border_size: size of padding on both sides of each dim
    :return:
    """
    all_coordinates = {}
    for name, image_shape in img_sizes.items():
        window_shape = win_sizes[name]
        grid_size = [max(win_size - 2 * border, 0)
                     for (win_size, border) in zip(window_shape, border_size)]
        assert len(image_shape) >= N_SPATIAL, \
            'incompatible image shapes in grid_spatial_coordinates'
        assert len(window_shape) >= N_SPATIAL, \
            'incompatible window shapes in grid_spatial_coordinates'
        assert len(grid_size) >= N_SPATIAL, \
            'incompatible border sizes in grid_spatial_coordinates'
        steps_along_each_dim = [
            _enumerate_step_points(starting=0,
                                   ending=image_shape[i],
                                   win_size=window_shape[i],
                                   step_size=grid_size[i])
            for i in range(N_SPATIAL)]
        starting_coords = np.asanyarray(np.meshgrid(*steps_along_each_dim))
        starting_coords = starting_coords.reshape((N_SPATIAL, -1)).T
        n_locations = starting_coords.shape[0]
        # prepare the output coordinates matrix
        spatial_coords = np.zeros((n_locations, N_SPATIAL * 2), dtype=np.int32)
        spatial_coords[:, :N_SPATIAL] = starting_coords
        for idx in range(N_SPATIAL):
            spatial_coords[:, N_SPATIAL + idx] = \
                starting_coords[:, idx] + window_shape[idx]
        max_coordinates = np.max(spatial_coords, axis=0)[N_SPATIAL:]
        assert np.all(max_coordinates <= image_shape[:N_SPATIAL]), \
            "window size greater than the spatial coordinates {} : {}".format(
                max_coordinates, image_shape)
        subject_list = np.ones((n_locations, 1), dtype=np.int32) * subject_id
        spatial_coords = np.append(subject_list, spatial_coords, axis=1)
        all_coordinates[name] = spatial_coords
    return all_coordinates


def _enumerate_step_points(starting, ending, win_size, step_size):
    """
    generate all possible sampling size in between starting and ending.

    :param starting: integer of starting value
    :param ending: integer of ending value
    :param win_size: integer of window length
    :param step_size: integer of distance between two sampling points
    :return: a set of unique sampling points
    """
    try:
        starting = max(int(starting), 0)
        ending = max(int(ending), 0)
        win_size = max(int(win_size), 1)
        step_size = max(int(step_size), 1)
    except (TypeError, ValueError):
        tf.logging.fatal(
            'step points should be specified by integers, received:'
            '%s, %s, %s, %s', starting, ending, win_size, step_size)
        raise ValueError
    if starting > ending:
        starting, ending = ending, starting
    sampling_point_set = []
    while (starting + win_size) <= ending:
        sampling_point_set.append(starting)
        starting = starting + step_size
    additional_last_point = ending - win_size
    sampling_point_set.append(max(additional_last_point, 0))
    sampling_point_set = np.unique(sampling_point_set).flatten()
    if len(sampling_point_set) == 2:
        # in case of too few samples, adding
        # an additional sampling point to
        # the middle between starting and ending
        sampling_point_set = np.append(
            sampling_point_set, np.round(np.mean(sampling_point_set)))
    _, uniq_idx = np.unique(sampling_point_set, return_index=True)
    return sampling_point_set[np.sort(uniq_idx)]

def zoom_3d(image, ratio, interp_order):
    """
    Taking 5D image as input, and zoom each 3D slice independently
    """
    assert image.ndim == 5, "input images should be 5D array"
    output = []
    for time_pt in range(image.shape[3]):
        output_mod = []
        for mod in range(image.shape[4]):
            zoomed = scipy.ndimage.zoom(
                image[..., time_pt, mod], ratio[:3], order=interp_order)
            output_mod.append(zoomed[..., np.newaxis, np.newaxis])
        output.append(np.concatenate(output_mod, axis=-1))
    return np.concatenate(output, axis=-2)

































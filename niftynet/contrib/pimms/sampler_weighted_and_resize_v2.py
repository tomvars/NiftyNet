# -*- coding: utf-8 -*-
"""
Generating uniformly distributed image window from input image
This can also be considered as a "random cropping" layer of the
input image.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from collections import OrderedDict
import scipy.ndimage
import tensorflow as tf
from tensorflow.python.data.util import nest

from niftynet.engine.image_window_dataset import ImageWindowDataset
from niftynet.contrib.csv_reader.sampler_csv_rows import ImageWindowDatasetCSV
from niftynet.contrib.csv_reader.csv_reader import apply_niftynet_format_to_data
from niftynet.engine.image_window import N_SPATIAL, LOCATION_FORMAT


class WeightedAndResizeSampler(ImageWindowDatasetCSV):
    """
    This class generates samples by uniformly sampling each input volume
    currently the coordinates are randomised for spatial dims only,
    i.e., the first three dims of image.

    This layer can be considered as a "random cropping" layer of the
    input image.
    """

    def __init__(self,
                 reader,
                 csv_reader=None,
                 window_sizes=None,
                 batch_size=1,
                 windows_per_image=1,
                 queue_length=10,
                 name='weighted_sampler_v2'):
        ImageWindowDatasetCSV.__init__(
            self,
            reader=reader,
            csv_reader=csv_reader,
            window_sizes=window_sizes,
            batch_size=batch_size,
            windows_per_image=windows_per_image,
            queue_length=queue_length,
            shuffle=True,
            epoch=-1,
            smaller_final_batch_mode='drop',
            name=name)

        tf.logging.info("initialised uniform sampler %s ", self.window.shapes)
        self.window_centers_sampler = weighted_spatial_coordinates

    # pylint: disable=too-many-locals
    def layer_op(self, idx=None):
        """
        This function generates sampling windows to the input buffer
        image data are from ``self.reader()``

        It first completes window shapes based on image data,
        then finds random coordinates based on the window shapes
        finally extract window with the coordinates and output
        a dictionary (required by input buffer).

        :return: output data dictionary
            ``{image_modality: data_array, image_location: n_samples * 7}``
        """
        image_id, data, interp_orders = self.reader(idx=idx, shuffle=True)
        ##### Randomly drop modalities according to params #####
        num_modalities = data['image'].shape[-1]
        # These probabilities are obtained using
        # from scipy.stats import t
        # N, rv = 4, t(0.1)
        # prob = [(rv.cdf((i+1)/N) - rv.cdf(i/N)) / (rv.cdf(1.0) - 0.5) for i in N]
        prob_dict = {3: [0.5078, 0.3014, 0.1908], 4: [0.4026, 0.2755, 0.1861, 0.1358]}
        modalities_to_drop = int(np.random.choice(range(num_modalities), 1, p=prob_dict[num_modalities]))
        data_shape_without_modality = list(data['image'].shape)[:-1]
        random_indices = np.random.permutation(range(num_modalities))
        dropped_indices = []
        for idx in range(modalities_to_drop):
            idx_to_drop = random_indices[idx]
            data['image'][..., idx_to_drop] = np.zeros(shape=data_shape_without_modality)
            dropped_indices.append(idx_to_drop)
        # Randomly permute the inputs
        permuted_indices = np.random.permutation(range(num_modalities))
        data['image'] = data['image'][..., permuted_indices]
        print('These are the modality shapes after random permuting', data['image'].shape)
        ########################################################
        # initialise output dict, placeholders as dictionary keys
        # this dictionary will be used in
        # enqueue operation in the form of: `feed_dict=output_dict`
        output_dict = {}
        # find random coordinates based on window and image shapes
        image_shapes = dict(
            (name, data[name].shape) for name in self.window.names)
        static_window_shapes = self.window.match_image_shapes(image_shapes)
        print('SHAPES', self.reader.get_subject_id(image_id), image_shapes, static_window_shapes, sep='\n')
        coordinates = self._spatial_coordinates_generator(
            subject_id=image_id,
            data=data,
            img_sizes=image_shapes,
            win_sizes=static_window_shapes,
            n_samples=self.window.n_samples)

        # fill output dict with data
        for name in list(data):
            coordinates_key = LOCATION_FORMAT.format(name)
            image_data_key = name

            # fill the coordinates
            location_array = coordinates[name]
            output_dict[coordinates_key] = location_array

            # fill output window array
            image_array = []
            for window_id in range(self.window.n_samples):
                x_start, y_start, z_start, x_end, y_end, z_end = \
                    location_array[window_id, 1:]
                try:
                    image_window = data[name][
                                   x_start:x_end, y_start:y_end, z_start:z_end, ...]
                    image_array.append(image_window[np.newaxis, ...])
                except ValueError:
                    tf.logging.fatal(
                        "dimensionality miss match in input volumes, "
                        "please specify spatial_window_size with a "
                        "3D tuple and make sure each element is "
                        "smaller than the image length in each dim. "
                        "Current coords %s", location_array[window_id])
                    raise
            if len(image_array) > 1:
                output_dict[image_data_key] = \
                    np.concatenate(image_array, axis=0)
            else:
                output_dict[image_data_key] = image_array[0]
        # the output image shape should be
        # [enqueue_batch_size, x, y, z, time, modality]
        # where enqueue_batch_size = windows_per_image
        if self.csv_reader is not None:
            _, label_dict, _ = self.csv_reader(idx=0)
            output_dict.update(label_dict)

            for name in self.csv_reader.names:
                output_dict[name + '_location'] = output_dict['image_location']
        ###### Update the output_dict with the permuted modalities ######
        output_dict['modality_label'] = apply_niftynet_format_to_data(np.tile(permuted_indices.astype(np.float32), 10))
        output_dict['modality_label_location'] = output_dict['image_location']
        #################################################################

        ################# RESIZING CENTRAL SLICE FOR USE BY MODALITY CLASSIFIER #################
        name = 'modality_slice'
        coordinates_key = LOCATION_FORMAT.format(name)
        image_data_key = name
        window_shape = (80, 80, 1, 1, 1)
        output_dict[coordinates_key] = self.dummy_coordinates(
            image_id, window_shape, self.window.n_samples).astype(np.int32)
        image_array = []
        for i in range(-5, 5):
            # prepare image data
            image_shape = tuple(list(data['image'].shape[:2]) + [1, 1, 1])
            if image_shape == window_shape or interp_orders['image'][0] < 0:
                # already in the same shape
                image_window = data['image']
            else:
                zoom_ratio = [float(p) / float(d) for p, d in zip(window_shape, image_shape)]
                central_slice = data['image'].shape[2] // 2
                image_window = zoom_3d(
                    image=data['image'][:, :, central_slice + i, ...][:, :, np.newaxis, ...],
                    # image=data['image'],
                    ratio=zoom_ratio,
                    interp_order=3)
            image_array.append(image_window[np.newaxis, ...])
        if len(image_array) > 1:
            output_dict[image_data_key] = np.concatenate(image_array, axis=3).astype(np.float32)
        else:
            output_dict[image_data_key] = image_array[0].astype(np.float32)
        # print('output_shape in weighted sampler', output_dict[image_data_key].shape)
        ##########################################################################################

        return output_dict

    def _spatial_coordinates_generator(self,
                                       subject_id,
                                       data,
                                       img_sizes,
                                       win_sizes,
                                       n_samples=1):
        """
        Generate spatial coordinates for sampling.

        Values in ``win_sizes`` could be different --
        for example in a segmentation network ``win_sizes`` could be
        ``{'training_image_spatial_window': (32, 32, 10),
           'Manual_label_spatial_window': (16, 16, 10)}``
        (the network reduces x-y plane spatial resolution).

        This function handles this situation by first find the largest
        window across these window definitions, and generate the coordinates.
        These coordinates are then adjusted for each of the
        smaller window sizes (the output windows are almost concentric).
        """

        assert data is not None, "No input from image reader. Please check" \
                                 "the configuration file."

        # infer the largest spatial window size and check image spatial shapes
        try:
            img_spatial_size, win_spatial_size = \
                _infer_spatial_size(img_sizes, win_sizes)
        except Exception as e:
            print(e)
            raise Exception('Failed to read with subject_id: {}'.format(self.reader.get_subject_id(subject_id)))

        sampling_prior_map = None
        try:
            sampling_prior_map = data.get('sampler', None)
        except AttributeError:
            pass

        n_samples = max(n_samples, 1)
        window_centres = self.window_centers_sampler(
            n_samples, img_spatial_size, win_spatial_size, sampling_prior_map)
        assert window_centres.shape == (n_samples, N_SPATIAL), \
            "the coordinates generator should return " \
            "{} samples of rank {} locations".format(n_samples, N_SPATIAL)

        # adjust spatial coordinates based on each mod spatial window size
        all_coordinates = {}
        for mod in list(win_sizes):
            win_size = np.asarray(win_sizes[mod][:N_SPATIAL])
            half_win = np.floor(win_size / 2.0).astype(int)

            # Make starting coordinates of the window
            spatial_coords = np.zeros(
                (n_samples, N_SPATIAL * 2), dtype=np.int32)
            spatial_coords[:, :N_SPATIAL] = np.maximum(
                window_centres[:, :N_SPATIAL] - half_win[:N_SPATIAL], 0)

            # Make the opposite corner of the window is
            # just adding the mod specific window size
            spatial_coords[:, N_SPATIAL:] = \
                spatial_coords[:, :N_SPATIAL] + win_size[:N_SPATIAL]
            assert np.all(spatial_coords[:, N_SPATIAL:] <= img_spatial_size), \
                'spatial coords: out of bounds.'

            # include subject id as the 1st column of all_coordinates values
            subject_id = np.ones((n_samples,), dtype=np.int32) * subject_id
            spatial_coords = np.append(
                subject_id[:, None], spatial_coords, axis=1)
            all_coordinates[mod] = spatial_coords

        return all_coordinates

    @property
    def tf_shapes(self):
        """
        returns a dictionary of sampler output tensor shapes
        """
        assert self.window, 'Unknown output shapes: self.window not initialised'
        shape_dict = self.window.tf_shapes
        if self.csv_reader is not None:
            shape_dict.update(self.csv_reader.tf_shapes)
        output_shapes = nest.map_structure_up_to(
            {'modality_slice': tf.float32,
             'modality_slice_location': tf.int32},
            tf.TensorShape,
            {'modality_slice': (1, 80, 80, 10, 1, 3),
             'modality_slice_location': (1, 7)}
        )
        shape_dict.update(output_shapes)
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
        shape_dict.update({'modality_slice': tf.float32,
                           'modality_slice_location': tf.int32})
        return shape_dict


def weighted_spatial_coordinates(
        n_samples, img_spatial_size, win_spatial_size, sampler_map):
    """
    Weighted sampling from a map.
    This function uses a cumulative histogram for fast sampling.

    see also `sampler_uniform.rand_spatial_coordinates`

    :param n_samples: number of random coordinates to generate
    :param img_spatial_size: input image size
    :param win_spatial_size: input window size
    :param sampler_map: sampling prior map, it's spatial shape should be
            consistent with `img_spatial_size`
    :return: (n_samples, N_SPATIAL) coordinates representing sampling
              window centres relative to img_spatial_size
    """
    assert sampler_map is not None, \
        'sampling prior map is not specified, ' \
        'please check `sampler=` option in the config.'
    # Get the cumulative sum of the normalised sorted intensities
    # i.e. first sort the sampling frequencies, normalise them
    # to sum to one, and then accumulate them in order
    assert np.all(img_spatial_size[:N_SPATIAL] ==
                  sampler_map.shape[:N_SPATIAL]), \
        'image and sampling map shapes do not match'
    win_spatial_size = np.asarray(win_spatial_size, dtype=np.int32)
    cropped_map = crop_sampling_map(sampler_map, win_spatial_size)
    flatten_map = cropped_map.flatten()
    flatten_map = flatten_map - np.min(flatten_map)
    normaliser = flatten_map.sum()
    # get the sorting indexes to that we can invert the sorting later on.
    sorted_indexes = np.argsort(flatten_map)
    sorted_data = np.cumsum(
        np.true_divide(flatten_map[sorted_indexes], normaliser))

    middle_coords = np.zeros((n_samples, N_SPATIAL), dtype=np.int32)
    for sample in range(0, n_samples):
        # get n_sample from the cumulative histogram, spaced by 1/n_samples,
        # plus a random perturbation to give us a stochastic sampler
        sample_ratio = 1 - (np.random.random() + sample) / (n_samples + 1)
        # find the index where the cumulative it above the sample threshold
        try:
            if normaliser == 0:
                # constant map? reducing to a uniform sampling
                sample_index = np.random.randint(len(sorted_data))
            else:
                sample_index = np.argmax(sorted_data >= sample_ratio)
        except ValueError:
            tf.logging.fatal("unable to choose sampling window based on "
                             "the current frequency map.")
            raise
        # invert the sample index to the pre-sorted index
        inverted_sample_index = sorted_indexes[sample_index]
        # get the x,y,z coordinates on the cropped_map
        middle_coords[sample, :N_SPATIAL] = np.unravel_index(
            inverted_sample_index, cropped_map.shape)[:N_SPATIAL]

    # re-shift coords due to the crop
    half_win = np.floor(win_spatial_size / 2).astype(np.int32)
    middle_coords[:, :N_SPATIAL] = \
        middle_coords[:, :N_SPATIAL] + half_win[:N_SPATIAL]
    return middle_coords


def crop_sampling_map(input_map, win_spatial_size):
    """
    Utility function for generating a cropped version of the
    input sampling prior map (the input weight map where the centre of
    the window might be). If the centre of the window was outside of
    this crop area, the patch would be outside of the field of view

    :param input_map: the input weight map where the centre of
                      the window might be
    :param win_spatial_size: size of the borders to be cropped
    :return: cropped sampling map
    """

    # prepare cropping indices
    _start, _end = [], []
    for win_size, img_size in \
            zip(win_spatial_size[:N_SPATIAL], input_map.shape[:N_SPATIAL]):
        # cropping floor of the half window
        d_start = int(win_size / 2.0)
        # using ceil of half window
        d_end = img_size - win_size + int(win_size / 2.0 + 0.6)

        _start.append(d_start)
        _end.append(d_end + 1 if d_start == d_end else d_end)

    try:
        assert len(_start) == 3
        cropped_map = input_map[
                      _start[0]:_end[0], _start[1]:_end[1], _start[2]:_end[2], 0, 0]
        assert np.all(cropped_map.shape) > 0
    except (IndexError, KeyError, TypeError, AssertionError):
        tf.logging.fatal(
            "incompatible map: %s and window size: %s\n"
            "try smaller (fully-specified) spatial window sizes?",
            input_map.shape, win_spatial_size)
        raise
    return cropped_map


def _infer_spatial_size(img_sizes, win_sizes):
    """
    Utility function to find the spatial size of image,
    and the largest spatial window size across input sections.

    Raises NotImplementedError if the images have
    different spatial dimensions.

    :param img_sizes: dictionary of {'input_name': (img_size_x, img_size,y,...)}
    :param win_sizes: dictionary of {'input_name': (win_size_x, win_size_y,...)}
    :return: (image_spatial_size, window_largest_spatial_size)
    """
    uniq_spatial_size = \
        set([img_size[:N_SPATIAL] for img_size in list(img_sizes.values())])
    if len(uniq_spatial_size) != 1:
        tf.logging.fatal("Don't know how to generate sampling "
                         "locations: Spatial dimensions of the "
                         "grouped input sources are not "
                         "consistent. %s", uniq_spatial_size)
        raise NotImplementedError
    img_spatial_size = np.asarray(uniq_spatial_size.pop(), dtype=np.int32)

    # find the largest spatial window across input sections
    _win_spatial_sizes = \
        [win_size[:N_SPATIAL] for win_size in win_sizes.values()]
    _win_spatial_sizes = np.asarray(_win_spatial_sizes, dtype=np.int32)
    win_spatial_size = np.max(_win_spatial_sizes, axis=0)

    assert all([img_spatial_size[i] >= win_spatial_size[i]
                for i in range(N_SPATIAL)]), \
        "window size {} is larger than image size {}".format(
            win_spatial_size, img_spatial_size)

    return img_spatial_size, win_spatial_size


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

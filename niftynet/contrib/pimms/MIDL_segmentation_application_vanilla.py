# -*- coding: utf-8 -*-
import tensorflow as tf
import math
from niftynet.application.base_application import BaseApplication
from niftynet.engine.application_factory import \
    ApplicationNetFactory, InitializerFactory, OptimiserFactory
from niftynet.engine.application_variables import \
    CONSOLE, NETWORK_OUTPUT, TF_SUMMARIES
from niftynet.contrib.csv_reader.sampler_grid_whole_volume_v2_csv import GridSampler
from niftynet.contrib.pimms.sampler_grid_whole_volume_v2_csv_midl import GridSampler as ValidationGridSampler

from niftynet.engine.sampler_resize_v2 import ResizeSampler
from niftynet.contrib.pimms.sampler_uniform_v2 import UniformSampler
from niftynet.contrib.pimms.sampler_weighted_v2 import WeightedSampler
# from niftynet.engine.sampler_weighted_v2 import WeightedSampler
from niftynet.engine.sampler_balanced_v2 import BalancedSampler
from niftynet.engine.windows_aggregator_grid import GridSamplesAggregator
# from niftynet.contrib.pimms.windows_aggregator_grid_multitask import GridSamplesAggregator
from niftynet.engine.windows_aggregator_resize import ResizeSamplesAggregator
from niftynet.io.image_reader import ImageReader
from niftynet.contrib.csv_reader.csv_reader import CSVReader
from niftynet.layer.binary_masking import BinaryMaskingLayer
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer
from niftynet.layer.loss_segmentation import LossFunction as LossFunctionSeg
from niftynet.layer.loss_regression import LossFunction as LossFunctionReg
from niftynet.layer.loss_classification import LossFunction as LossFunctionClass
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.layer.pad import PadLayer
from niftynet.layer.post_processing import PostProcessingLayer
from niftynet.layer.rand_flip import RandomFlipLayer
from niftynet.layer.rand_rotation import RandomRotationLayer
from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer
from niftynet.evaluation.segmentation_evaluator import SegmentationEvaluator
from niftynet.layer.rand_elastic_deform import RandomElasticDeformationLayer
from niftynet.contrib.pimms.multitask_hemis_3D_shared_weights import MultitaskHeMIS3D
# from niftynet.contrib.pimms.multitask_hemis_3D_unet import MultitaskHeMIS3D

SUPPORTED_INPUT = set(['image', 'label', 'modality_label', 'weight', 'sampler', 'inferred'])


class SegmentationApplication(BaseApplication):
    REQUIRED_CONFIG_SECTION = "SEGMENTATION"

    def __init__(self, net_param, action_param, action):
        super(SegmentationApplication, self).__init__()
        tf.logging.info('starting segmentation application')
        self.action = action

        self.net_param = net_param
        self.action_param = action_param

        self.data_param = None
        self.segmentation_param = None
        self.use_dataset_a_value = True
        self.SUPPORTED_SAMPLING = {
            'uniform': (self.initialise_uniform_sampler,
                        self.initialise_grid_sampler,
                        self.initialise_grid_aggregator),
            'weighted': (self.initialise_weighted_sampler,
                         self.initialise_grid_sampler,
                         self.initialise_grid_aggregator),
            'resize': (self.initialise_resize_sampler,
                       self.initialise_resize_sampler,
                       self.initialise_resize_aggregator),
            'balanced': (self.initialise_balanced_sampler,
                         self.initialise_grid_sampler,
                         self.initialise_grid_aggregator),
        }

    def initialise_dataset_loader(
            self, data_param=None, task_param=None, data_partitioner=None):

        self.data_param = data_param
        self.segmentation_param = task_param

        # initialise input image readers
        if self.is_training:
            reader_names = ('image', 'label', 'weight', 'sampler')
            csv_reader_names = ('modality_label',)
            # csv_reader_names = ('',)
        elif self.is_inference:
            # in the inference process use `image` input only
            reader_names = ('image',)
            csv_reader_names = ('modality_label',)
        elif self.is_evaluation:
            reader_names = ('image', 'label', 'inferred')
        else:
            tf.logging.fatal(
                'Action `%s` not supported. Expected one of %s',
                self.action, self.SUPPORTED_PHASES)
            raise ValueError
        try:
            reader_phase = self.action_param.dataset_to_infer
        except AttributeError:
            reader_phase = None
        file_lists = data_partitioner.get_file_lists_by(
            phase=reader_phase, action=self.action)
        self.csv_readers = [
            CSVReader(csv_reader_names).initialise(
                data_param, task_param, file_list) for file_list in file_lists]
        self.readers = [
            ImageReader(reader_names).initialise(
                data_param, task_param, file_list) for file_list in file_lists]
        # initialise input preprocessing layers
        foreground_masking_layer = BinaryMaskingLayer(
            type_str=self.net_param.foreground_type,
            multimod_fusion=self.net_param.multimod_foreground_type,
            threshold=0.0) \
            if self.net_param.normalise_foreground_only else None
        mean_var_normaliser = MeanVarNormalisationLayer(
            image_name='image', binary_masking_func=foreground_masking_layer) \
            if self.net_param.whitening else None
        histogram_normaliser = HistogramNormalisationLayer(
            image_name='image',
            modalities=vars(task_param).get('image'),
            model_filename=self.net_param.histogram_ref_file,
            binary_masking_func=foreground_masking_layer,
            norm_type=self.net_param.norm_type,
            cutoff=self.net_param.cutoff,
            name='hist_norm_layer') \
            if (self.net_param.histogram_ref_file and
                self.net_param.normalisation) else None
        label_normalisers = None
        if self.net_param.histogram_ref_file and \
                task_param.label_normalisation:
            print(vars(task_param).get('label'))
            label_normalisers = [DiscreteLabelNormalisationLayer(
                image_name='label',
                modalities=('Parcellation',),
                model_filename=self.net_param.histogram_ref_file,
                num_threads=self.net_param.num_threads
            )]
            if self.is_evaluation:
                label_normalisers.append(
                    DiscreteLabelNormalisationLayer(
                        image_name='inferred',
                        modalities=vars(task_param).get('inferred'),
                        model_filename=self.net_param.histogram_ref_file))
                label_normalisers[-1].key = label_normalisers[0].key

        normalisation_layers = []
        if histogram_normaliser is not None:
            normalisation_layers.append(histogram_normaliser)
        if mean_var_normaliser is not None:
            normalisation_layers.append(mean_var_normaliser)
        if task_param.label_normalisation and \
                (self.is_training or not task_param.output_prob):
            normalisation_layers.extend(label_normalisers)

        volume_padding_layer = []
        if self.net_param.volume_padding_size:
            volume_padding_layer.append(PadLayer(
                image_name=SUPPORTED_INPUT,
                border=self.net_param.volume_padding_size,
                mode=self.net_param.volume_padding_mode))

        # initialise training data augmentation layers
        augmentation_layers = []
        if self.is_training:
            train_param = self.action_param
            if train_param.random_flipping_axes != -1:
                augmentation_layers.append(RandomFlipLayer(
                    flip_axes=train_param.random_flipping_axes))
            if train_param.scaling_percentage:
                augmentation_layers.append(RandomSpatialScalingLayer(
                    min_percentage=train_param.scaling_percentage[0],
                    max_percentage=train_param.scaling_percentage[1]))
            if train_param.rotation_angle or \
                    train_param.rotation_angle_x or \
                    train_param.rotation_angle_y or \
                    train_param.rotation_angle_z:
                rotation_layer = RandomRotationLayer()
                if train_param.rotation_angle:
                    rotation_layer.init_uniform_angle(
                        train_param.rotation_angle)
                else:
                    rotation_layer.init_non_uniform_angle(
                        train_param.rotation_angle_x,
                        train_param.rotation_angle_y,
                        train_param.rotation_angle_z)
                augmentation_layers.append(rotation_layer)
            if train_param.do_elastic_deformation:
                spatial_rank = list(self.readers[0].spatial_ranks.values())[0]
                augmentation_layers.append(RandomElasticDeformationLayer(
                    spatial_rank=spatial_rank,
                    num_controlpoints=train_param.num_ctrl_points,
                    std_deformation_sigma=train_param.deformation_sigma,
                    proportion_to_augment=train_param.proportion_to_deform))

        # only add augmentation to first reader (not validation reader)
        self.readers[0].add_preprocessing_layers(
            volume_padding_layer + normalisation_layers + augmentation_layers)

        for reader in self.readers[1:]:
            reader.add_preprocessing_layers(
                volume_padding_layer + normalisation_layers)

    def initialise_uniform_sampler(self):
        self.sampler = [[UniformSampler(
            reader=self.readers[0],
            csv_reader=self.csv_readers[0],
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            queue_length=self.net_param.queue_length),
            GridSampler(
                reader=self.readers[1],
                csv_reader=self.csv_readers[1],
                window_sizes=self.data_param,
                batch_size=self.net_param.batch_size,
                spatial_window_size=self.action_param.spatial_window_size,
                window_border=self.action_param.border,
                smaller_final_batch_mode=self.net_param.smaller_final_batch_mode,
                queue_length=self.net_param.queue_length
            ) if self.action_param.do_whole_volume_validation else UniformSampler(
                reader=self.readers[0],
                csv_reader=self.csv_readers[0],
                window_sizes=self.data_param,
                batch_size=self.net_param.batch_size,
                windows_per_image=self.action_param.sample_per_volume,
                queue_length=self.net_param.queue_length)
        ]]

    def initialise_weighted_sampler(self):
        self.sampler = [[WeightedSampler(
            reader=self.readers[0],
            csv_reader=self.csv_readers[0],
            mode=None,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            queue_length=self.net_param.queue_length),
            GridSampler(
                reader=self.readers[1],
                csv_reader=self.csv_readers[1],
                window_sizes=self.data_param,
                batch_size=self.net_param.batch_size,
                spatial_window_size=self.action_param.spatial_window_size,
                window_border=self.action_param.border,
                smaller_final_batch_mode=self.net_param.smaller_final_batch_mode,
                queue_length=self.net_param.queue_length
            ) if self.action_param.do_whole_volume_validation else WeightedSampler(
                reader=self.readers[0],
                csv_reader=self.csv_readers[0],
                mode=None,
                window_sizes=self.data_param,
                batch_size=self.net_param.batch_size,
                windows_per_image=self.action_param.sample_per_volume,
                queue_length=self.net_param.queue_length)
        ]]

    def initialise_resize_sampler(self):
        self.sampler = [[ResizeSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            shuffle=self.is_training,
            smaller_final_batch_mode=self.net_param.smaller_final_batch_mode,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_grid_sampler(self):
        print(self.segmentation_param.idxs_to_drop)
        self.sampler = [[ValidationGridSampler(
            reader=reader,
            csv_reader=csv_reader,
            window_sizes=self.data_param,
            idxs_to_drop=self.segmentation_param.idxs_to_drop,
            batch_size=self.net_param.batch_size,
            spatial_window_size=self.action_param.spatial_window_size,
            window_border=self.action_param.border,
            smaller_final_batch_mode=self.net_param.smaller_final_batch_mode,
            queue_length=self.net_param.queue_length
        ) for reader, csv_reader in zip(self.readers, self.csv_readers)]]

    def initialise_balanced_sampler(self):
        self.sampler = [[BalancedSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_grid_aggregator(self):
        self.output_decoder_lesion = GridSamplesAggregator(
            image_reader=self.readers[0],
            output_path=self.action_param.save_seg_dir,
            window_border=self.action_param.border,
            interp_order=self.action_param.output_interp_order,
            postfix='_lesion_niftynet_out'
        )
        self.output_decoder_tumour = GridSamplesAggregator(
            image_reader=self.readers[0],
            output_path=self.action_param.save_seg_dir,
            window_border=self.action_param.border,
            interp_order=self.action_param.output_interp_order,
            postfix='_tumour_niftynet_out'
        )
        self.output_decoder_parcellation = GridSamplesAggregator(
            image_reader=self.readers[0],
            output_path=self.action_param.save_seg_dir,
            window_border=self.action_param.border,
            interp_order=self.action_param.output_interp_order,
            postfix='_parcellation_niftynet_out'
        )

    def initialise_resize_aggregator(self):
        self.output_decoder = ResizeSamplesAggregator(
            image_reader=self.readers[0],
            output_path=self.action_param.save_seg_dir,
            window_border=self.action_param.border,
            interp_order=self.action_param.output_interp_order,
            postfix=self.action_param.output_postfix)

    def initialise_sampler(self):
        if self.is_training:
            self.SUPPORTED_SAMPLING[self.net_param.window_sampling][0]()
        elif self.is_inference:
            self.SUPPORTED_SAMPLING[self.net_param.window_sampling][1]()

    def initialise_aggregator(self):
        self.SUPPORTED_SAMPLING[self.net_param.window_sampling][2]()

    def initialise_network(self):
        w_regularizer = None
        b_regularizer = None
        reg_type = self.net_param.reg_type.lower()
        decay = self.net_param.decay
        if reg_type == 'l2' and decay > 0:
            from tensorflow.contrib.layers.python.layers import regularizers
            w_regularizer = regularizers.l2_regularizer(decay)
            b_regularizer = regularizers.l2_regularizer(decay)
        elif reg_type == 'l1' and decay > 0:
            from tensorflow.contrib.layers.python.layers import regularizers
            w_regularizer = regularizers.l1_regularizer(decay)
            b_regularizer = regularizers.l1_regularizer(decay)

        self.net = MultitaskHeMIS3D(
            n_modalities=4,
            num_classes=self.segmentation_param.num_classes,
            w_initializer=InitializerFactory.get_initializer(
                name=self.net_param.weight_initializer),
            b_initializer=InitializerFactory.get_initializer(
                name=self.net_param.bias_initializer),
            w_regularizer=w_regularizer,
            b_regularizer=b_regularizer,
            acti_func=self.net_param.activation_function)

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):

        def switch_sampler(for_training):
            with tf.name_scope('train' if for_training else 'validation'):
                sampler = self.get_sampler()[0][0 if for_training else -1]
                return sampler.pop_batch_op()
        if self.is_training:
            if self.action_param.validation_every_n > 0:
                data_dict = tf.cond(tf.logical_not(self.is_validation),
                                    lambda: switch_sampler(for_training=True),
                                    lambda: switch_sampler(for_training=False))
            else:
                data_dict = switch_sampler(for_training=True)

            image = tf.cast(data_dict['image'], tf.float32)
            net_args = {'is_training': self.is_training}
            net_out, tumour_out, brain_parcellation, class_out, brain_parcellation_activation, _, _ = self.net(image,
                                                                                                  **net_args)

            with tf.name_scope('Optimiser'):
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.action_param.lr)

            ### All the dims in the loss functions
            tf.logging.info(data_dict.get('label', None))
            tf.logging.info(data_dict.get('label', None)[..., 0])
            tf.logging.info(data_dict.get('label', None)[..., 1])
            tf.logging.info(net_out)
            tf.logging.info(brain_parcellation)

            seg_loss_func = LossFunctionSeg(
                n_class=2,
                loss_type=self.action_param.loss_type,
                softmax=self.segmentation_param.softmax)

            lesion_seg_loss = seg_loss_func(
                prediction=net_out,
                ground_truth=data_dict.get('label', None)[..., 0],
                weight_map=data_dict.get('weight', None))

            seg_loss_func = LossFunctionSeg(
                n_class=4,
                loss_type=self.action_param.loss_type,
                softmax=self.segmentation_param.softmax)

            tumour_seg_loss = seg_loss_func(
                prediction=tumour_out,
                ground_truth=data_dict.get('label', None)[..., 1],
                weight_map=data_dict.get('weight', None))

            brain_parcellation_loss_func = LossFunctionSeg(
                n_class=160,
                loss_type=self.action_param.loss_type,
                softmax=self.segmentation_param.softmax)

            brain_parcellation_loss = brain_parcellation_loss_func(
                prediction=brain_parcellation,
                ground_truth=data_dict.get('label', None)[..., 2],
                weight_map=tf.to_float(tf.less(data_dict.get('label', None)[..., 1], 1)))#tf.to_float(tf.less(data_dict.get('label', None)[..., 1], 1)))

            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            lesion_load = tf.count_nonzero(data_dict.get('label', None)[..., 0])
            if self.net_param.decay > 0.0 and reg_losses:
                reg_loss = tf.reduce_mean(
                    [tf.reduce_mean(reg_loss) for reg_loss in reg_losses])
                # tumour_loss = tumour_seg_loss + reg_loss + lesion_seg_loss
                # lesion_loss = lesion_seg_loss + brain_parcellation_loss + reg_loss + tumour_seg_loss
                # loss = tf.cond(is_lesion,
                #                true_fn=lambda: tumour_loss,
                #                false_fn=lambda: lesion_loss)
                loss = tumour_seg_loss + lesion_seg_loss + brain_parcellation_loss + reg_loss
            else:
                # tumour_loss = tumour_seg_loss + lesion_seg_loss
                # lesion_loss = lesion_seg_loss + brain_parcellation_loss + tumour_seg_loss
                # loss = tf.cond(is_lesion,
                #                true_fn=lambda: tumour_loss,
                #                false_fn=lambda: lesion_loss)
                loss = tumour_seg_loss + lesion_seg_loss + brain_parcellation_loss
            grads = self.optimiser.compute_gradients(
                loss, colocate_gradients_with_ops=True)
            # collecting gradients variables
            gradients_collector.add_to_collection([grads])
            # collecting output variables

            def my_tf_round(x, decimals=0):
                multiplier = tf.constant(10 ** decimals, dtype=x.dtype)
                return tf.round(x * multiplier) / multiplier

            outputs_collector.add_to_collection(
                var=my_tf_round(loss, 4), name='loss',
                average_over_devices=False, collection=CONSOLE)

            outputs_collector.add_to_collection(
                var=my_tf_round(lesion_seg_loss, 4), name='lesion_seg_loss',
                average_over_devices=False, collection=CONSOLE)

            outputs_collector.add_to_collection(
                var=my_tf_round(tumour_seg_loss, 4), name='tumour_seg_loss',
                average_over_devices=False, collection=CONSOLE)

            outputs_collector.add_to_collection(
                var=my_tf_round(brain_parcellation_loss, 4), name='parcellation_loss',
                average_over_devices=False, collection=CONSOLE)

            outputs_collector.add_to_collection(
                var=loss, name='loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

            outputs_collector.add_to_collection(
                var=lesion_seg_loss, name='lesion_seg_loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

            outputs_collector.add_to_collection(
                var=tumour_seg_loss, name='tumour_seg_loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

            outputs_collector.add_to_collection(
                var=brain_parcellation_loss, name='brain_parcellation_loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

            # look at only one modality
            for idx, modality in enumerate(['T1', 'T1c', 'T2', 'FLAIR']):
                outputs_collector.add_to_collection(
                    var=tf.contrib.image.rotate(255.0 * (image[:1, ..., idx] - tf.reduce_min(image[:1, ..., idx])) /
                                                (tf.reduce_max(image[:1, ..., idx] - tf.reduce_min(image[:1, ..., idx]))),
                                                3 * math.pi / 2), name=modality,
                    average_over_devices=True, summary_type='image3_axial',
                    collection=TF_SUMMARIES)
            brain_parcellation = tf.argmax(brain_parcellation, axis=-1)
            outputs_collector.add_to_collection(
                var=tf.contrib.image.rotate(255 * (brain_parcellation[:1, ...] - tf.reduce_min(brain_parcellation[:1, ...])) / (
                    tf.reduce_max(brain_parcellation[:1, ...] - tf.reduce_min(brain_parcellation[:1, ...]))),
                                            3 * math.pi / 2), name='brain_parcellation',
                average_over_devices=True, summary_type='image3_axial',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=tf.contrib.image.rotate(
                    255 * (data_dict.get('label', None)[..., 2] - tf.reduce_min(data_dict.get('label', None)[..., 2])) /
                    (tf.reduce_max(data_dict.get('label', None)[..., 2] - tf.reduce_min(data_dict.get('label', None)[..., 2]))),
                    3 * math.pi / 2), name='brain_parcellation_gt',
                average_over_devices=True, summary_type='image3_axial',
                collection=TF_SUMMARIES)
            net_out = tf.argmax(net_out, axis=-1)
            outputs_collector.add_to_collection(
                var=tf.contrib.image.rotate(255 * (net_out[:1, ...]), 3 * math.pi / 2), name='lesion_segmentation',
                average_over_devices=True, summary_type='image3_axial',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=tf.contrib.image.rotate(
                    255 * (data_dict.get('label', None)[..., 0]),
                    3 * math.pi / 2), name='lesion_segmentation_gt',
                average_over_devices=True, summary_type='image3_axial',
                collection=TF_SUMMARIES)
            net_out = tf.argmax(tumour_out, axis=-1)[:1, ...]
            outputs_collector.add_to_collection(
                var=tf.contrib.image.rotate(
                    255 * (net_out - tf.reduce_min(net_out)) /
                    (tf.reduce_max(net_out - tf.reduce_min(net_out))),
                    3 * math.pi / 2), name='tumour_segmentation',
                average_over_devices=True, summary_type='image3_axial',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=tf.contrib.image.rotate(
                    255 * (data_dict.get('label', None)[..., 1] - tf.reduce_min(data_dict.get('label', None)[..., 1])) /
                    (tf.reduce_max(data_dict.get('label', None)[..., 1] - tf.reduce_min(data_dict.get('label', None)[..., 1]))),
                    3 * math.pi / 2), name='tumour_segmentation_gt',
                average_over_devices=True, summary_type='image3_axial',
                collection=TF_SUMMARIES)

        elif self.is_inference:
            # converting logits into final output for
            # classification probabilities or argmax classification labels
            data_dict = self.get_sampler()[0][0].pop_batch_op()
            image = tf.cast(data_dict['image'], tf.float32)
            net_args = {'is_training': True}
            net_out, tumour_out, brain_parcellation, class_out, brain_parcellation_activation, _, _ = self.net(image,
                                                                                                         **net_args)

            output_prob = self.segmentation_param.output_prob
            num_classes = self.segmentation_param.num_classes
            if output_prob and num_classes > 1:
                post_process_layer = PostProcessingLayer(
                    'SOFTMAX', num_classes=num_classes)
            elif not output_prob and num_classes > 1:
                post_process_layer = PostProcessingLayer(
                    'ARGMAX', num_classes=num_classes)
            else:
                post_process_layer = PostProcessingLayer(
                    'IDENTITY', num_classes=num_classes)
            net_out = post_process_layer(net_out)
            brain_parcellation = post_process_layer(brain_parcellation)
            tumour_out = post_process_layer(tumour_out)

            outputs_collector.add_to_collection(
                var=net_out, name='window',
                average_over_devices=False, collection=NETWORK_OUTPUT)
            outputs_collector.add_to_collection(
                var=tumour_out, name='tumour',
                average_over_devices=False, collection=NETWORK_OUTPUT)
            outputs_collector.add_to_collection(
                var=brain_parcellation, name='brain_parcellation',
                average_over_devices=False, collection=NETWORK_OUTPUT)
            outputs_collector.add_to_collection(
                var=data_dict['image_location'], name='location',
                average_over_devices=False, collection=NETWORK_OUTPUT)
            self.initialise_aggregator()

    def interpret_output(self, batch_output):
        if self.is_inference:
            end_signal_lesion = self.output_decoder_lesion.decode_batch(
                batch_output['window'], batch_output['location'])
            end_signal_parcellation =self.output_decoder_parcellation.decode_batch(
                batch_output['brain_parcellation'], batch_output['location'])
            end_signal_tumour = self.output_decoder_tumour.decode_batch(
                batch_output['tumour'], batch_output['location'])
            print(end_signal_tumour, end_signal_parcellation, end_signal_lesion)
            if all([not end_signal_lesion, not end_signal_parcellation, not end_signal_tumour]):
                return False
        return True

    def initialise_evaluator(self, eval_param):
        self.eval_param = eval_param
        self.evaluator = SegmentationEvaluator(self.readers[0],
                                               self.segmentation_param,
                                               eval_param)

    def add_inferred_output(self, data_param, task_param):
        return self.add_inferred_output_like(data_param, task_param, 'label')

    def set_iteration_update(self, iteration_message):
        """
        At each iteration ``application_driver`` calls::

            output = tf.session.run(variables_to_eval, feed_dict=data_dict)

        to evaluate TF graph elements, where
        ``variables_to_eval`` and ``data_dict`` are retrieved from
        ``iteration_message.ops_to_run`` and
        ``iteration_message.data_feed_dict``
         (In addition to the variables collected by self.output_collector).

        The output of `tf.session.run(...)` will be stored at
        ``iteration_message.current_iter_output``, and can be accessed
        from ``engine.handler_network_output.OutputInterpreter``.

        override this function for more complex operations
        (such as learning rate decay) according to
        ``iteration_message.current_iter``.
        """
        if iteration_message.is_training:
            iteration_message.data_feed_dict[self.is_validation] = False
        elif iteration_message.is_validation:
            iteration_message.data_feed_dict[self.is_validation] = True

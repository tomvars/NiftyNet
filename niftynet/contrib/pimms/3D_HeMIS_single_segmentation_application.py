# -*- coding: utf-8 -*-
import tensorflow as tf
import math
from niftynet.application.base_application import BaseApplication
from niftynet.engine.application_factory import \
    ApplicationNetFactory, InitializerFactory, OptimiserFactory
from niftynet.engine.application_variables import \
    CONSOLE, NETWORK_OUTPUT, TF_SUMMARIES
from niftynet.contrib.csv_reader.sampler_grid_whole_volume_v2_csv import GridSampler as ValidationGridSampler
from niftynet.engine.sampler_grid_v2 import GridSampler as GridSampler
from niftynet.engine.sampler_resize_v2 import ResizeSampler
from niftynet.contrib.pimms.sampler_uniform_v2 import UniformSampler
from niftynet.engine.sampler_weighted_v2 import WeightedSampler
from niftynet.engine.sampler_balanced_v2 import BalancedSampler
from niftynet.engine.windows_aggregator_grid import GridSamplesAggregator
from niftynet.engine.windows_aggregator_resize import ResizeSamplesAggregator
from niftynet.io.image_reader import ImageReader
from niftynet.contrib.csv_reader.csv_reader import CSVReader
from niftynet.layer.binary_masking import BinaryMaskingLayer
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer
from niftynet.layer.loss_segmentation import LossFunction as LossFunctionSeg
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
            csv_reader_names = ('',)
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
                modalities=('Parcellation'),
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
            ValidationGridSampler(
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
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

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
        self.sampler = [[GridSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            spatial_window_size=self.action_param.spatial_window_size,
            window_border=self.action_param.border,
            smaller_final_batch_mode=self.net_param.smaller_final_batch_mode,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_balanced_sampler(self):
        self.sampler = [[BalancedSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_grid_aggregator(self):
        self.output_decoder = GridSamplesAggregator(
            image_reader=self.readers[0] if self.is_inference else self.readers[1],
            output_path=self.action_param.save_seg_dir,
            window_border=self.action_param.border,
            interp_order=self.action_param.output_interp_order,
            postfix=self.action_param.output_postfix)

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

        self.net = ApplicationNetFactory.create(self.net_param.name)(
            n_modalities=self.readers[0].shapes['image'][-1],
            num_classes=self.segmentation_param.num_classes,
            w_initializer=InitializerFactory.get_initializer(
                name=self.net_param.weight_initializer),
            b_initializer=InitializerFactory.get_initializer(
                name=self.net_param.bias_initializer),
            w_regularizer=w_regularizer,
            b_regularizer=b_regularizer,
            acti_func=self.net_param.activation_function)

    def add_confusion_matrix_summaries_(self,
                                        num_classes,
                                        outputs_collector,
                                        class_out,
                                        data_dict):
        """ This method defines several monitoring metrics that
        are derived from the confusion matrix """
        labels = tf.reshape(tf.cast(data_dict['modality_label'], tf.int64), [-1])
        prediction = tf.reshape(tf.argmax(class_out, -1), [-1])
        conf_mat = tf.confusion_matrix(labels, prediction, num_classes)
        conf_mat = tf.to_float(conf_mat)
        TP, FP, TN, FN = conf_mat[1][1], conf_mat[0][1], conf_mat[0][0], conf_mat[1][0]
        if num_classes == 2:
            outputs_collector.add_to_collection(
                var=TP, name='true_positives',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=FN, name='false_negatives',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=FP, name='false_positives',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=FN, name='true_negatives',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

        outputs_collector.add_to_collection(
            var=(TP + TN) / (TP + FP + TN + FN), name='accuracy',
            average_over_devices=True, summary_type='scalar',
            collection=TF_SUMMARIES)

        outputs_collector.add_to_collection(
            var=(TP + TN) / (TP + FP + TN + FN), name='accuracy',
            average_over_devices=True, summary_type='scalar',
            collection=CONSOLE)

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):

        def switch_sampler(for_training):
            with tf.name_scope('train' if for_training else 'validation'):
                sampler = self.get_sampler()[0][0 if for_training else -1]
                return sampler.pop_batch_op()

        if self.is_training:

            current_iter = tf.placeholder(dtype=tf.float32)
            outputs_collector.add_to_collection(
                var=current_iter, name='current_iter',
                average_over_devices=False, collection=NETWORK_OUTPUT)
            if self.action_param.validation_every_n > 0:
                data_dict = tf.cond(tf.logical_not(self.is_validation),
                                    lambda: switch_sampler(for_training=True),
                                    lambda: switch_sampler(for_training=False))
            else:
                data_dict = switch_sampler(for_training=True)

            image = tf.cast(data_dict['image'], tf.float32)
            net_args = {'is_training': self.is_training}
            net_out, class_out = self.net(image, **net_args)

            with tf.name_scope('Optimiser'):
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.action_param.lr)

            seg_loss_func = LossFunctionSeg(
                n_class=self.segmentation_param.num_classes,
                loss_type=self.action_param.loss_type,
                softmax=self.segmentation_param.softmax)

            seg_loss = seg_loss_func(
                prediction=net_out,
                ground_truth=data_dict.get('label', None)[..., 0],
                weight_map=data_dict.get('weight', None))

            classification_loss_func = LossFunctionClass(
                n_class=image.shape.as_list()[-1],
                loss_type='CrossEntropy',
                multilabel=False
            )
            modality_classification_loss = classification_loss_func(
                prediction=tf.reshape(class_out, [-1, image.shape.as_list()[-1]]),
                ground_truth=data_dict.get('modality_label', None)
            )
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if self.net_param.decay > 0.0 and reg_losses:
                reg_loss = tf.reduce_mean(
                    [tf.reduce_mean(reg_loss) for reg_loss in reg_losses])
                loss = seg_loss + reg_loss + modality_classification_loss
            else:
                loss = seg_loss + modality_classification_loss
            grads = self.optimiser.compute_gradients(
                loss, colocate_gradients_with_ops=True)
            # collecting gradients variables
            gradients_collector.add_to_collection([grads])
            # collecting output variables

            outputs_collector.add_to_collection(
                var=loss, name='loss',
                average_over_devices=False, collection=CONSOLE)

            outputs_collector.add_to_collection(
                var=seg_loss, name='seg_loss',
                average_over_devices=False, collection=CONSOLE)

            dice_score_func = LossFunctionSeg(
                n_class=self.segmentation_param.num_classes,
                loss_type='Dice',
                softmax=self.segmentation_param.softmax)

            dice_score = dice_score_func(
                prediction=net_out,
                ground_truth=data_dict.get('label', None)[..., 0],
                weight_map=data_dict.get('weight', None))

            outputs_collector.add_to_collection(
                var=1 - dice_score, name='dice_score',
                average_over_devices=False, collection=CONSOLE)

            outputs_collector.add_to_collection(
                var=1 - dice_score, name='dice_score', summary_type='scalar',
                average_over_devices=False, collection=TF_SUMMARIES)

            outputs_collector.add_to_collection(
                var=modality_classification_loss,
                name='modality_classification_loss',
                average_over_devices=False, collection=CONSOLE)

            outputs_collector.add_to_collection(
                var=class_out, name='class_out',
                average_over_devices=False, collection=CONSOLE)

            outputs_collector.add_to_collection(
                var=loss, name='loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

            outputs_collector.add_to_collection(
                var=seg_loss, name='seg_loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

            outputs_collector.add_to_collection(
                var=modality_classification_loss, name='modality_classification_loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

            # look at only one modality
            num_modalities = image.shape.as_list()[-1]
            dict_of_modality_lists = {3: ['Flair', 'T1', 'T2'], 4: ['T1', 'T1c', 'T2', 'Flair']}
            for idx, modality in enumerate(dict_of_modality_lists[num_modalities]):
                outputs_collector.add_to_collection(
                    var=tf.contrib.image.rotate(255.0 * (image[:1, ..., idx] - tf.reduce_min(image[:1, ..., idx])) /
                                                (tf.reduce_max(image[:1, ..., idx] - tf.reduce_min(image[:1, ..., idx]))),
                                                3 * math.pi / 2), name=modality,
                    average_over_devices=True, summary_type='image3_axial',
                    collection=TF_SUMMARIES)
            net_out = tf.argmax(net_out, axis=-1)
            outputs_collector.add_to_collection(
                var=tf.contrib.image.rotate(255 * (net_out[:1, ...]), 3 * math.pi / 2), name='segmentation',
                average_over_devices=True, summary_type='image3_axial',
                collection=TF_SUMMARIES)
            label_tensor = data_dict.get('label', None)[:1, ..., 0]
            outputs_collector.add_to_collection(
                var=tf.contrib.image.rotate(255 * label_tensor, 3 * math.pi / 2), name='segmentation_gt',
                average_over_devices=True, summary_type='image3_axial',
                collection=TF_SUMMARIES)

            self.add_confusion_matrix_summaries_(num_classes=image.shape.as_list()[-1],
                                                 outputs_collector=outputs_collector,
                                                 class_out=class_out,
                                                 data_dict=data_dict)

        elif self.is_inference:
            # converting logits into final output for
            # classification probabilities or argmax classification labels
            data_dict = switch_sampler(for_training=False)
            current_iter = tf.placeholder(dtype=tf.float32)
            outputs_collector.add_to_collection(
                var=current_iter, name='current_iter',
                average_over_devices=False, collection=NETWORK_OUTPUT)

            image = tf.cast(data_dict['image'], tf.float32)
            net_args = {'is_training': self.is_training}
            net_out, class_out = self.net(image, **net_args)

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

            outputs_collector.add_to_collection(
                var=net_out, name='window',
                average_over_devices=False, collection=NETWORK_OUTPUT)
            outputs_collector.add_to_collection(
                var=data_dict['image_location'], name='location',
                average_over_devices=False, collection=NETWORK_OUTPUT)
            self.initialise_aggregator()

    def interpret_output(self, batch_output):
        return self.output_decoder.decode_batch(batch_output['window'],
                                                batch_output['location'])

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
        iteration_message.data_feed_dict[iteration_message.ops_to_run['niftynetout']['current_iter']] = iteration_message.current_iter

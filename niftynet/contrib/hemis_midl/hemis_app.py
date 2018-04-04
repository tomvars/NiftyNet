import tensorflow as tf
import sys
import os
import numpy as np
sys.path.append(os.getcwd())

from niftynet.application.base_application import BaseApplication
from niftynet.engine.application_factory import \
    ApplicationNetFactory, OptimiserFactory
from niftynet.engine.application_variables import \
    CONSOLE, NETWORK_OUTPUT, TF_SUMMARIES
from niftynet.engine.sampler_grid import GridSampler
from niftynet.contrib.hemis_midl.hemis_sampler import HeMISSampler
from niftynet.engine.windows_aggregator_grid import GridSamplesAggregator
from niftynet.io.image_reader import ImageReader
from niftynet.layer.binary_masking import BinaryMaskingLayer
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer
from niftynet.layer.loss_segmentation import LossFunction as LossFunction_Seg
from niftynet.layer.loss_classification import LossFunction as LossFunction_Classification
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.layer.pad import PadLayer
from niftynet.layer.post_processing import PostProcessingLayer
from niftynet.layer.loss_segmentation import dice_dense, dice_nosquare

SUPPORTED_INPUT = set(['image', 'label', 'modalities'])


class BRATSApp(BaseApplication):
    REQUIRED_CONFIG_SECTION = "MULTITASK"

    def __init__(self, net_param, action_param, action):
        BaseApplication.__init__(self)
        tf.logging.info('starting BRATS segmentation app')
        self.action = action

        self.net_param = net_param
        self.action_param = action_param

        self.data_param = None
        self.segmentation_param = None
        self.current_iter = None

    def initialise_dataset_loader(
            self, data_param=None, task_param=None, data_partitioner=None):
        self.data_param = data_param
        self.segmentation_param = task_param

        # read each line of csv files into an instance of Subject
        if self.is_training:
            file_lists = []
            if self.action_param.validation_every_n > 0:
                file_lists.append(data_partitioner.train_files)
                file_lists.append(data_partitioner.validation_files)
            else:
                file_lists.append(data_partitioner.all_files)
            self.readers = []
            for file_list in file_lists:
                reader = ImageReader(SUPPORTED_INPUT)
                reader.initialise(data_param, task_param, file_list)
                self.readers.append(reader)
        else:  # in the inference process use image input only
            inference_reader = ImageReader(['image'])
            file_list = data_partitioner.inference_files
            inference_reader.initialise(data_param, task_param, file_list)
            self.readers = [inference_reader]

        foreground_masking_layer = None
        if self.net_param.normalise_foreground_only:
            foreground_masking_layer = BinaryMaskingLayer(
                type_str=self.net_param.foreground_type,
                multimod_fusion=self.net_param.multimod_foreground_type,
                threshold=0.0)

        mean_var_normaliser = MeanVarNormalisationLayer(
            image_name='image', binary_masking_func=foreground_masking_layer)

        label_normaliser = DiscreteLabelNormalisationLayer(
            image_name='label',
            modalities=vars(task_param).get('label'),
            model_filename=self.net_param.histogram_ref_file)

        normalisation_layers = []
        normalisation_layers.append(mean_var_normaliser)
        if task_param.label_normalisation:
            normalisation_layers.append(label_normaliser)

        volume_padding_layer = []
        if self.net_param.volume_padding_size:
            volume_padding_layer.append(PadLayer(
                image_name=SUPPORTED_INPUT,
                border=self.net_param.volume_padding_size))
        for reader in self.readers:
            reader.add_preprocessing_layers(
                normalisation_layers + volume_padding_layer)

    def initialise_sampler(self):
        if self.is_training:
            print('Initialising HeMIS sampler with {}'.format(len(self.segmentation_param.image)))
            self.sampler = [[HeMISSampler(
                reader=reader,
                data_param=self.data_param,
                batch_size=self.net_param.batch_size,
                windows_per_image=self.action_param.sample_per_volume,
                queue_length=self.net_param.queue_length) for reader in
                self.readers]]
        else:
            self.sampler = [[HeMISSampler(
                reader=reader,
                data_param=self.data_param,
                batch_size=self.net_param.batch_size,
                windows_per_image=self.action_param.sample_per_volume,
                queue_length=self.net_param.queue_length) for reader in
                self.readers]]

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
        print(w_regularizer, b_regularizer)
        self.net = ApplicationNetFactory.create(self.net_param.name)(
            num_classes=self.segmentation_param.num_classes_seg,
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
                                    lambda: switch_sampler(True),
                                    lambda: switch_sampler(False))
            else:
                data_dict = switch_sampler(for_training=True)

            image = tf.cast(data_dict['image'], tf.float32)
            current_iter = tf.placeholder(dtype=tf.float32, name='global_step')
            (net_out, modality_scores) = self.net(image, is_training=self.is_training, outputs_collector=outputs_collector)
            with tf.name_scope('Optimiser'):
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.action_param.lr)
            #### Add a classification loss! #####
            loss_func_classification = LossFunction_Classification(
                n_class=self.segmentation_param.num_classes_classification,
                loss_type="CrossEntropy"
            )
            loss_func_seg = LossFunction_Seg(
                n_class=self.segmentation_param.num_classes_seg,
                loss_type=self.action_param.loss_type)
            tf.logging.info('Size of net_out %s' % net_out.shape)
            ground_truth = data_dict.get('label', None)
            tf.logging.info('net_out %s' % net_out.shape)
            tf.logging.info('ground_truth %s' % ground_truth.shape)
            modality_ground_truth = data_dict.get('modalities', None)
            tf.logging.info('modality_ground_truth %s' % modality_ground_truth.shape)
            tf.logging.info('modality_scores %s' % modality_scores.shape)

            data_loss_seg = loss_func_seg(
                prediction=net_out,
                ground_truth=ground_truth,
                weight_map=data_dict.get('weight', None)
            )

            data_loss_classification = loss_func_classification(
                prediction=modality_scores,
                ground_truth=modality_ground_truth
            )
            reg_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)
            outputs_collector.add_to_collection(
                var=current_iter, name='iter',
                average_over_devices=True, summary_type='scalar',
                collection=NETWORK_OUTPUT)
            decay_constant = self.segmentation_param.decay_constant
            class_loss_multiplier = tf.exp(-decay_constant*tf.cast(current_iter, dtype=tf.float32), name='class_loss_multiplier')
            if self.net_param.decay > 0.0 and reg_losses:
                reg_loss = tf.reduce_mean(
                    [tf.reduce_mean(reg_loss) for reg_loss in reg_losses])
                loss = data_loss_seg + reg_loss + (data_loss_classification * class_loss_multiplier)
            else:
                loss = data_loss_seg
            # outputs_collector.add_to_collection(
            #     var=class_loss_multiplier, name='class_loss_multiplier',
            #     average_over_devices=True, summary_type='scalar',
            #     collection=CONSOLE)
            #
            # outputs_collector.add_to_collection(
            #     var=class_loss_multiplier, name='class_loss_multiplier',
            #     average_over_devices=True, summary_type='scalar',
            #     collection=TF_SUMMARIES)
            # with tf.name_scope('Optimiser_Classification'):
            #     optimiser_class = OptimiserFactory.create(name="adam")
            #     classification_optimizer = optimiser_class.get_instance(learning_rate=3e-4)
            grads_seg = self.optimiser.compute_gradients(loss)
            # grads_classification = classification_optimizer.compute_gradients(data_loss_classification)
            # collecting gradients variables
            # grads_classification = [x for x in grads_classification if 'modality_classifier' in x[1].name]
            # grads_seg = [x for x in grads_seg if 'modality_classifier' not in x[1].name]
            # print('grads_classification', grads_classification)
            # print('grads_seg', grads_seg)
            gradients_collector.add_to_collection([grads_seg])
            # collecting output variables
            tf.logging.info('net_out %s' % net_out.shape)
            tf.logging.info('ground_truth %s' % ground_truth.shape)
            tf.logging.info('image %s' % image.shape)
            one_hot_ground_truth = tf.squeeze(
                tf.one_hot(
                    tf.cast(ground_truth, tf.uint8), 2
                )
            )
            tf.logging.info('one_hot_ground_truth %s' % one_hot_ground_truth.shape)
            # one_hot_ground_truth = tf.reverse(one_hot_ground_truth, axis=[-1])
            tf.logging.info('one_hot_ground_truth %s' % one_hot_ground_truth.shape)

            outputs_collector.add_to_collection(
                var=data_loss_seg, name='dice_loss_seg',
                average_over_devices=True, collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=data_loss_seg, name='dice_loss_seg',
                average_over_devices=True, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=data_loss_classification, name='class_loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=data_loss_classification, name='class_loss',
                average_over_devices=True, summary_type='scalar',
                collection=CONSOLE)

            # outputs_collector.add_to_collection(
            #     var=modality_ground_truth, name='modality_ground_truth',
            #     average_over_devices=True, summary_type='scalar',
            #     collection=CONSOLE)
            #
            # outputs_collector.add_to_collection(
            #     var=modality_scores, name='modality_scores',
            #     average_over_devices=True, summary_type='scalar',
            #     collection=CONSOLE)

            def dice_nosquare(prediction, one_hot):
                """
                Function to calculate the classical dice loss

                :param prediction: the logits
                :param ground_truth: the segmentation ground_truth
                :param weight_map:
                :return: the loss
                """

                dice_numerator = 2.0 * tf.reduce_sum(one_hot * prediction, reduction_indices=[0])
                dice_denominator = tf.reduce_sum(prediction, reduction_indices=[0]) + \
                                   tf.reduce_sum(one_hot, reduction_indices=[0])
                epsilon_denominator = 0.00001

                dice_score = dice_numerator / (dice_denominator + epsilon_denominator)
                # dice_score.set_shape([n_classes])
                # minimising (1 - dice_coefficients)
                return tf.reduce_mean(dice_score)

            outputs_collector.add_to_collection(
                var=dice_nosquare(prediction=net_out, one_hot=one_hot_ground_truth), name='dice_score',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=dice_nosquare(prediction=net_out, one_hot=one_hot_ground_truth), name='dice_score',
                average_over_devices=True, summary_type='scalar',
                collection=CONSOLE)

            #Flair,T1,T1c,T2 whatever is in the SEGMENTATION param
            for i, name in enumerate(self.segmentation_param.image):
                outputs_collector.add_to_collection(
                    var=tf.expand_dims(image[:, :, :, i], -1), name=name,
                    average_over_devices=True, summary_type='image',
                    collection=TF_SUMMARIES)

            outputs_collector.add_to_collection(
                var=ground_truth, name='ground_truth',
                average_over_devices=True, summary_type='image',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=tf.expand_dims(net_out[:, :, :, 1], -1), name='net_out',
                average_over_devices=True, summary_type='image',
                collection=TF_SUMMARIES)

            def add_confusion_matrix_summaries_(outputs_collector,
                                                prediction,
                                                labels):
                """ This method defines several monitoring metrics that
                are derived from the confusion matrix """
                print(prediction)
                print(labels)
                labels = tf.reshape(labels, [-1])
                prediction = tf.reshape(tf.argmax(prediction, -1), [-1])
                num_classes = self.segmentation_param.num_classes_classification
                conf_mat = tf.contrib.metrics.confusion_matrix(labels,
                                                               prediction,
                                                               num_classes)
                conf_mat = tf.to_float(conf_mat) / float(self.net_param.batch_size)
                if num_classes == 2:
                    outputs_collector.add_to_collection(
                        var=conf_mat[1][1], name='true_positives',
                        average_over_devices=True, summary_type='scalar',
                        collection=TF_SUMMARIES)
                    outputs_collector.add_to_collection(
                        var=conf_mat[1][0], name='false_negatives',
                        average_over_devices=True, summary_type='scalar',
                        collection=TF_SUMMARIES)
                    outputs_collector.add_to_collection(
                        var=conf_mat[0][1], name='false_positives',
                        average_over_devices=True, summary_type='scalar',
                        collection=TF_SUMMARIES)
                    outputs_collector.add_to_collection(
                        var=conf_mat[0][0], name='true_negatives',
                        average_over_devices=True, summary_type='scalar',
                        collection=TF_SUMMARIES)
                else:
                    outputs_collector.add_to_collection(
                        var=conf_mat[tf.newaxis, :, :, tf.newaxis],
                        name='confusion_matrix',
                        average_over_devices=True, summary_type='image',
                        collection=TF_SUMMARIES)
                total_zero_inputs = tf.reduce_sum(conf_mat[0, :])
                correct_zero_inputs = conf_mat[0][0]
                outputs_collector.add_to_collection(
                    var= correct_zero_inputs / total_zero_inputs,
                    name='class_acc_zero_inputs',
                    average_over_devices=True, summary_type='scalar',
                    collection=TF_SUMMARIES)

                outputs_collector.add_to_collection(
                    var=correct_zero_inputs / total_zero_inputs,
                    name='class_acc_zero_inputs',
                    average_over_devices=True, summary_type='scalar',
                    collection=CONSOLE)

                outputs_collector.add_to_collection(
                    var=(tf.trace(conf_mat) - conf_mat[0][0])/(tf.reduce_sum(conf_mat, axis=[0,1]) - total_zero_inputs),
                    name='class_acc_non_zero_inputs',
                    average_over_devices=True, summary_type='scalar',
                    collection=TF_SUMMARIES)

                outputs_collector.add_to_collection(
                    var=(tf.trace(conf_mat) - conf_mat[0][0]) / (
                                tf.reduce_sum(conf_mat, axis=[0, 1]) - total_zero_inputs),
                    name='class_acc_non_zero_inputs',
                    average_over_devices=True, summary_type='scalar',
                    collection=CONSOLE)

            add_confusion_matrix_summaries_(outputs_collector=outputs_collector,
                                            prediction=modality_scores,
                                            labels=modality_ground_truth)

            outputs_collector.add_to_collection(
                var=modality_scores[0],
                name='modality_scores',
                average_over_devices=True, summary_type='scalar',
                collection=CONSOLE)

            outputs_collector.add_to_collection(
                var=modality_ground_truth[0][0],
                name='modality_ground_truth',
                average_over_devices=True, summary_type='scalar',
                collection=CONSOLE)
        else:
            # converting logits into final output for
            # classification probabilities or argmax classification labels
            data_dict = switch_sampler(for_training=False)
            image = tf.cast(data_dict['image'], tf.float32)
            (net_out, modality_scores) = self.net(image, is_training=self.is_training)

            output_prob = self.segmentation_param.output_prob
            num_classes = self.segmentation_param.num_classes_seg
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

            self.output_decoder = GridSamplesAggregator(
                image_reader=self.readers[0],
                output_path=self.action_param.save_seg_dir,
                window_border=self.action_param.border,
                interp_order=self.action_param.output_interp_order)

    def interpret_output(self, batch_output):
        if not self.is_training:
            return self.output_decoder.decode_batch(
                batch_output['window'], batch_output['location'])
        return True

    def set_iteration_update(self, iteration_message):
        """
        At each iteration ``application_driver`` calls:
            ``output = tf.session.run(variables_to_eval, feed_dict=data_dict)``

        to evaluate TF graph elements, where
        ``variables_to_eval`` and ``data_dict`` are retrieved from
        ``application_iteration.IterationMessage.ops_to_run`` and
        ``application_iteration.IterationMessage.data_feed_dict``.
        in addition to the variables collected by output_collector;
        implemented in ``application_driver.run_vars``)

        This function (is called before ``tf.session.run`` by the
        driver) provides an interface for accessing ``variables_to_eval`` and
        ``data_dict`` at each iteration.

        Override this function for more complex operations according to
        ``application_iteration.IterationMessage.current_iter``.
        """
        self.current_iter = iteration_message.current_iter
        if iteration_message.is_training:
            iteration_message.data_feed_dict[self.is_validation] = False
        elif iteration_message.is_validation:
            iteration_message.data_feed_dict[self.is_validation] = True

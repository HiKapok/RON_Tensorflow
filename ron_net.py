# Copyright 2016 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a RON model using a given dataset."""
import tensorflow as tf
import os

from tensorflow.python import debug as tf_debug
from tensorflow.python.ops import control_flow_ops

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import tf_utils

#import logging

slim = tf.contrib.slim

# # get TF logger
# log = logging.getLogger('tensorflow')

# # create formatter and add it to the handlers
# formatter = logging.Formatter('%(asctime)s: %(levelname)s %(name)s - %(message)s')
# log.setFormatter(formatter)



DATA_FORMAT = 'NHWC' #'NCHW'

# =========================================================================== #
# RON Network flags.
# =========================================================================== #

tf.app.flags.DEFINE_float(
    'loss_alpha', 1./3, 'Alpha parameter in the loss function.')
tf.app.flags.DEFINE_float(
    'loss_beta', 1./3, 'Beta parameter in the loss function.')
tf.app.flags.DEFINE_float(
    'negative_ratio', 3., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float(
    'match_threshold', 0.5, 'Matching threshold in the loss function.')
tf.app.flags.DEFINE_float(
    'neg_threshold', 0.3, 'Matching threshold for the negtive examples in the loss function.')
tf.app.flags.DEFINE_float(
    'objectness_thres', 0.03, 'threshold for the objectness to indicate the exist of object in that location.')
# =========================================================================== #
# General Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'model_dir', './logs/',
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer(
    'num_readers', 12,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 24,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'num_cpu_threads', 64,
    'The number of cpu cores used to train.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 7200,
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1., 'GPU memory fraction to use.')

# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'weight_decay', 0.0005, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'optimizer', 'momentum',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.00001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.5,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', 'pascalvoc_0712', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'data_dir', '../PASCAL/VOC_TF/VOC0712TF/', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
    'model_name', 'ron_320_vgg', 'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'batch_size', 24, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'checkpoint_path', None, #'../vgg_model/vgg16_reducedfc.ckpt',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', 'vgg_16',#None,
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', 'ron_320_vgg/reverse_module, ron_320_vgg/conv6, ron_320_vgg/conv7',#None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', True, #False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS

# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
    if not FLAGS.data_dir:
        raise ValueError('You must supply the dataset directory with --data_dir')

    tf.logging.set_verbosity(tf.logging.DEBUG)
    #with tf.Graph().as_default():
    # with tf.Graph().as_default(), tf.device('/cpu:0'):
    with tf.device('/cpu:0'):
        global_step = slim.create_global_step()

        # Select the dataset.
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.data_dir)

        # Get the RON network and its anchors.
        ron_class = nets_factory.get_network(FLAGS.model_name)
        ron_params = ron_class.default_params._replace(num_classes=FLAGS.num_classes)
        ron_net = ron_class(ron_params)
        ron_shape = ron_net.params.img_shape
        ron_anchors = ron_net.anchors(ron_shape)

        # for i in range(len(ron_anchors)):
        #     for j in range(len(ron_anchors[i])):
        #         print(ron_anchors[i][j].shape)

        tf_utils.print_configuration(FLAGS.__flags, ron_params,
                                     dataset.data_sources, FLAGS.model_dir)
        # =================================================================== #
        # Create a dataset provider and batches.
        # =================================================================== #
        with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=FLAGS.num_readers,
                common_queue_capacity=120 * FLAGS.batch_size,
                common_queue_min=80 * FLAGS.batch_size,
                shuffle=True)
        # Get for RON network: image, labels, bboxes.
        # (ymin, xmin, ymax, xmax) fro gbboxes
        [image, shape, glabels, gbboxes, isdifficult] = provider.get(['image', 'shape',
                                                         'object/label',
                                                         'object/bbox',
                                                         'object/difficult'])
        glabels = tf.cast(isdifficult < tf.ones_like(isdifficult), glabels.dtype) * glabels
        #### DEBUG ####
        #image = tf.Print(image, [shape, glabels, gbboxes], message='before preprocess: ', summarize=20)
        # Select the preprocessing function.
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training=True)

        # Pre-processing image, labels and bboxes.
        image, glabels, gbboxes = image_preprocessing_fn(image, glabels, gbboxes,
                                   out_shape=ron_shape,
                                   data_format=DATA_FORMAT)

        #### DEBUG ####
        #image = tf.Print(image, [shape, glabels, gbboxes], message='after preprocess: ', summarize=20)

        #glabels = tf.Print(glabels, [glabels], message='glabels: ', summarize=20)


        # Encode groundtruth labels and bboxes.
        # glocalisations is our regression object
        # gclasses is the ground_trutuh label
        # gscores is the the jaccard score with ground_truth
        gclasses, glocalisations, gscores = \
            ron_net.bboxes_encode(glabels, gbboxes, ron_anchors, positive_threshold=FLAGS.match_threshold, ignore_threshold=FLAGS.neg_threshold)

        # each size of the batch elements
        # include one image, three others(gclasses, glocalisations, gscores)
        batch_shape = [1] + [len(ron_anchors)] * 3

        # Training batches and queue.
        r = tf.train.batch(
            tf_utils.reshape_list([image, gclasses, glocalisations, gscores]),
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=120 * FLAGS.batch_size)
        b_image, b_gclasses, b_glocalisations, b_gscores = \
            tf_utils.reshape_list(r, batch_shape)

        with tf.device('/gpu:0'):
            # Construct RON network.
            arg_scope = ron_net.arg_scope(weight_decay=FLAGS.weight_decay,
                                          data_format=DATA_FORMAT)
            with slim.arg_scope(arg_scope):
                predictions, logits, objness_pred, objness_logits, localisations, end_points = \
                    ron_net.net(b_image, is_training=True)
            # Add loss function.
            ron_net.losses(logits, localisations, objness_logits, objness_pred,
                           b_gclasses, b_glocalisations, b_gscores,
                           match_threshold = FLAGS.match_threshold,
                           neg_threshold = FLAGS.neg_threshold,
                           objness_threshold = FLAGS.objectness_thres,
                           negative_ratio=FLAGS.negative_ratio,
                           alpha=FLAGS.loss_alpha,
                           beta=FLAGS.loss_beta,
                           label_smoothing=FLAGS.label_smoothing)

            # Gather initial summaries.
            summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            # Add summaries for losses and extra losses.
            for loss in tf.get_collection(tf.GraphKeys.LOSSES):
                summaries.add(tf.summary.scalar(loss.op.name, loss))
            for loss in tf.get_collection('EXTRA_LOSSES'):
                summaries.add(tf.summary.scalar(loss.op.name, loss))

            # =================================================================== #
            # Configure the moving averages.
            # =================================================================== #
            if FLAGS.moving_average_decay:
                moving_average_variables = slim.get_model_variables()
                variable_averages = tf.train.ExponentialMovingAverage(
                    FLAGS.moving_average_decay, global_step)
            else:
                moving_average_variables, variable_averages = None, None

            # =================================================================== #
            # Configure the optimization procedure.
            # =================================================================== #
            # learning_rate = tf_utils.configure_learning_rate(FLAGS,
            #                                                  dataset.num_samples,
            #                                                  global_step)

            lr_values = [FLAGS.learning_rate * decay for decay in [1., 0.1, 0.001]]
            learning_rate_ = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), [90000, 115000], lr_values)
            learning_rate = tf.maximum(learning_rate_, tf.constant(FLAGS.end_learning_rate, dtype=learning_rate_.dtype))


            optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)

            if FLAGS.moving_average_decay:
                # Update ops executed locally by trainer.
                update_ops.append(variable_averages.apply(moving_average_variables))
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

            # Variables to train.
            variables_to_train = tf_utils.get_variables_to_train(FLAGS)

            # and returns a train_tensor and summary_op
            total_loss = tf.losses.get_total_loss()
            # Add total_loss to summary.
            summaries.add(tf.summary.scalar('total_loss', total_loss))

            # Create gradient updates.
            grads = optimizer.compute_gradients(
                                    total_loss,
                                    variables_to_train)

            grad_updates = optimizer.apply_gradients(grads,
                                                     global_step=global_step)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops)
            train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                              name='train_op')

            # Merge all summaries together.
            summary_op = tf.summary.merge(list(summaries), name='summary_op')

            # =================================================================== #
            # Kicks off the training.
            # =================================================================== #
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction)
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement = False, intra_op_parallelism_threads = FLAGS.num_cpu_threads, inter_op_parallelism_threads = FLAGS.num_cpu_threads, gpu_options = gpu_options)
            saver = tf.train.Saver(max_to_keep=5,
                                   keep_checkpoint_every_n_hours = FLAGS.save_interval_secs/3600.,
                                   write_version=2,
                                   pad_step_number=False)
            def wrapper_debug(sess):
                sess = tf_debug.LocalCLIDebugWrapperSession(sess, thread_name_filter="MainThread$")
                sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
                return sess

            slim.learning.train(
                train_tensor,
                logdir=FLAGS.model_dir,
                master='',
                is_chief=True,
                init_fn=tf_utils.get_init_fn(FLAGS, os.path.join(FLAGS.data_dir, 'vgg_model/vgg16_reducedfc.ckpt')),
                summary_op=summary_op,
                number_of_steps=FLAGS.max_number_of_steps,
                log_every_n_steps=FLAGS.log_every_n_steps,
                save_summaries_secs=FLAGS.save_summaries_secs,
                saver=saver,
                save_interval_secs=FLAGS.save_interval_secs,
                session_config=config,
                session_wrapper=None,#wrapper_debug,#
                sync_optimizer=None)


if __name__ == '__main__':
    tf.app.run()

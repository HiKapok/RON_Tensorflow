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

from replicate_model_fn import replicate_model_fn
from replicate_model_fn import TowerOptimizer
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
    'num_readers', 8,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 16,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'num_cpu_threads', 0,
    'The number of cpu cores used to train.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summaries_steps', 500,
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
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
# gradients in replicate_model_fn are sumed in multi-GPU mode
tf.app.flags.DEFINE_float('learning_rate', 0.0013, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.92, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'decay_steps', 1000,
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
    'batch_size', 32, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')
# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'checkpoint_path', None, #'./checkpoints/ssd_300_vgg.ckpt',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', 'vgg_16',#None,
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', 'ron_320_vgg/reverse_module',#None,
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

def validate_batch_size_for_multi_gpu(batch_size):
    """For multi-gpu, batch-size must be a multiple of the number of
    available GPUs.

    Note that this should eventually be handled by replicate_model_fn
    directly. Multi-GPU support is currently experimental, however,
    so doing the work here until that feature is in place.
    """
    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()
    num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
    num_gpus=1
    if not num_gpus:
        raise ValueError('Multi-GPU mode was specified, but no GPUs '
                        'were found. To use CPU, run without --multi_gpu.')

    remainder = batch_size % num_gpus
    if remainder:
        err = ('When running with multiple GPUs, batch size '
                'must be a multiple of the number of available GPUs. '
                'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
                ).format(num_gpus, batch_size, batch_size - remainder)
        raise ValueError(err)


def get_init_fn_for_scaffold(extra_path):
    if FLAGS.checkpoint_path is None:
        flags_checkpoint_path = extra_path
    else:
        flags_checkpoint_path = FLAGS.checkpoint_path
    # Warn the user if a checkpoint exists in the model_dir. Then ignore.
    if tf.train.latest_checkpoint(FLAGS.model_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % FLAGS.model_dir)
        return None
    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    # Change model scope if necessary.
    if FLAGS.checkpoint_model_scope is not None:
        variables_to_restore = \
            {var.op.name.replace(FLAGS.model_name,
                                 FLAGS.checkpoint_model_scope): var
             for var in variables_to_restore}

    if tf.gfile.IsDirectory(flags_checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(flags_checkpoint_path)
    else:
        checkpoint_path = flags_checkpoint_path

    tf.logging.info('Fine-tuning from %s. Ignoring missing vars: %s' % (checkpoint_path, FLAGS.ignore_missing_vars))

    if not variables_to_restore:
            raise ValueError('variables_to_restore cannot be empty')
    if FLAGS.ignore_missing_vars:
        reader = tf.train.NewCheckpointReader(checkpoint_path)
        if isinstance(variables_to_restore, dict):
            var_dict = variables_to_restore
        else:
            var_dict = {var.op.name: var for var in variables_to_restore}
        available_vars = {}
        for var in var_dict:
            if reader.has_tensor(var):
                available_vars[var] = var_dict[var]
            else:
                tf.logging.warning('Variable %s missing in checkpoint %s', var, checkpoint_path)
        variables_to_restore = available_vars
    if variables_to_restore:
        saver = tf.train.Saver(variables_to_restore, reshape=False)
        saver.build()
        def callback(scaffold, session):
            saver.restore(session, checkpoint_path)
        return callback
    else:
        tf.logging.warning('No Variables to restore')
        return None


def model_fn(ron_net, image, gclasses, glocalisations, gscores, mode, params):
    """The model_fn argument for creating an Estimator."""
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                              global_step,
                                              FLAGS.decay_steps,
                                              FLAGS.learning_rate_decay_factor,
                                              staircase=True,
                                              name='exponential_decay_learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(tf.maximum(learning_rate, tf.constant(FLAGS.end_learning_rate, dtype=learning_rate.dtype)), momentum=FLAGS.momentum, name='MomentumOptimizer')
        # If we are running multi-GPU, we need to wrap the optimizer.
        optimizer = TowerOptimizer(optimizer)

        arg_scope = ron_net.arg_scope(weight_decay=FLAGS.weight_decay,
                                              data_format=DATA_FORMAT)
        with slim.arg_scope(arg_scope):
            _, logits, objness_pred, objness_logits, localisations, _ = ron_net.net(image, is_training=True)
        # Add loss function.
        ron_net.losses(logits, localisations, objness_logits, objness_pred,
                       gclasses, glocalisations, gscores,
                       match_threshold = FLAGS.match_threshold,
                       neg_threshold = FLAGS.neg_threshold,
                       objness_threshold = FLAGS.objectness_thres,
                       negative_ratio=FLAGS.negative_ratio,
                       alpha=FLAGS.loss_alpha,
                       beta=FLAGS.loss_beta,
                       label_smoothing=FLAGS.label_smoothing)

        loss = tf.losses.get_total_loss()

        tf.identity(loss, name='loss_to_log')
        tf.identity(learning_rate, name='learning_rate_to_log')
        tf.identity(global_step, name='global_step_to_log')

        tf.summary.scalar('total_loss', loss)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, global_step),
            scaffold = tf.train.Scaffold(init_fn=get_init_fn_for_scaffold(os.path.join(FLAGS.data_dir, 'vgg_16.ckpt'))))
    raise ValueError('This Model Function Only Support Training Now!')
# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
    if not FLAGS.data_dir:
        raise ValueError('You must supply the dataset directory with --data_dir')

    tf.logging.set_verbosity(tf.logging.INFO)

    validate_batch_size_for_multi_gpu(FLAGS.batch_size)

    # Get the RON network and its anchors.
    ron_class = nets_factory.get_network(FLAGS.model_name)
    ron_params = ron_class.default_params._replace(num_classes=FLAGS.num_classes)
    ron_net = ron_class(ron_params)
    ron_shape = ron_net.params.img_shape
    ron_anchors = ron_net.anchors(ron_shape)
    # There are two steps required if using multi-GPU: (1) wrap the model_fn,
    # and (2) wrap the optimizer. The first happens here, and (2) happens
    # in the model_fn itself when the optimizer is defined.
    model_function = replicate_model_fn(lambda features, labels, mode, params, config: model_fn(ron_net, features, labels['b_gclasses'], labels['b_glocalisations'], labels['b_gscores'], mode, params), loss_reduction=tf.losses.Reduction.MEAN)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement = False, intra_op_parallelism_threads = FLAGS.num_cpu_threads, inter_op_parallelism_threads = FLAGS.num_cpu_threads, gpu_options = gpu_options)


    ron_detector = tf.estimator.Estimator(
      model_fn=model_function,
      model_dir=FLAGS.model_dir,
      params=None,
      config = tf.estimator.RunConfig(save_summary_steps = FLAGS.save_summaries_steps,
                                    save_checkpoints_secs = FLAGS.save_interval_secs,
                                    session_config = config,
                                    keep_checkpoint_max = 5,
                                    keep_checkpoint_every_n_hours = FLAGS.save_interval_secs/3600.,
                                    log_step_count_steps = FLAGS.log_every_n_steps))

    # Train the model
    def train_input_fn():
        # Select the dataset.
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.data_dir)
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
        # Select the preprocessing function.
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training=True)

        # Pre-processing image, labels and bboxes.
        image, glabels, gbboxes = image_preprocessing_fn(image, glabels, gbboxes,
                                   out_shape=ron_shape,
                                   data_format=DATA_FORMAT)

        # Encode groundtruth labels and bboxes.
        # glocalisations is our regression object
        # gclasses is the ground_trutuh label
        # gscores is the the jaccard score with ground_truth
        gclasses, glocalisations, gscores = ron_net.bboxes_encode(glabels, gbboxes, ron_anchors, positive_threshold=FLAGS.match_threshold, ignore_threshold=FLAGS.neg_threshold)

        # each size of the batch elements
        # include one image, three others(gclasses, glocalisations, gscores)
        batch_shape = [1] + [len(ron_anchors)] * 3

        # Training batches and queue.
        r = tf.train.batch(
            tf_utils.reshape_list([image, gclasses, glocalisations, gscores]),
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=120 * FLAGS.batch_size,
            shared_name=None)
        b_image, b_gclasses, b_glocalisations, b_gscores = tf_utils.reshape_list(r, batch_shape)
        return b_image, {'b_gclasses':b_gclasses, 'b_glocalisations':b_glocalisations, 'b_gscores':b_gscores}

    # Set up training hook that logs the training accuracy every 100 steps.
    tensors_to_log = {'total_loss': 'loss_to_log',
                    'learning_rate': 'learning_rate_to_log',
                    'global_step': 'global_step_to_log'}

    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps)
    #with tf.contrib.tfprof.ProfileContext('./train_dir') as pctx:
    ron_detector.train(input_fn=train_input_fn, hooks=[logging_hook], max_steps=FLAGS.max_number_of_steps)

if __name__ == '__main__':
    tf.app.run()

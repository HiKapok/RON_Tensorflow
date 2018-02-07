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
from tensorflow.python.platform import tf_logging
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
    'num_readers', 16,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 38,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 7200,
    'The frequency with which the model is saved, in seconds.')

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
tf.app.flags.DEFINE_float('learning_rate', 0.0012, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.00008,
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
    'data_dir', '../PASCAL/tfrecords/VOC0712', 'The directory where the dataset files are stored.')
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
    'batch_size', 4, 'The number of samples in each batch.')
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

# =========================================================================== #
# Multi-GPU training Flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task performs the variable "
                     "initialization ")
tf.app.flags.DEFINE_integer("num_gpus", 4,
                     "Total number of gpus for each machine worker."
                     "If you don't use GPU, please set it to '0'")
# for sync_replicas mode only
# when in async mode, we update params for each received gradients
tf.app.flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
tf.app.flags.DEFINE_string("ps_hosts", "localhost:2222",
                    "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", None, "job name: worker or ps")

FLAGS = tf.app.flags.FLAGS

def average_gradients(tower_grads):
    average_grads = []
    #tower_grads = [[(grad0_gpu0, var0_gpu0), (grad1_gpu0, var1_gpu0)], [(grad0_gpu1, var0_gpu1), (grad1_gpu1, var1_gpu1)]]
    #zip(*tower_grads)] = [((grad0_gpu0, var0_gpu0), (grad0_gpu1, var0_gpu1)), ((grad1_gpu0, var1_gpu0), (grad1_gpu1, var1_gpu1))]
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
    if not FLAGS.data_dir:
        raise ValueError('You must supply the dataset directory with --data_dir')
    num_gpus = FLAGS.num_gpus
    if num_gpus < 1: num_gpus = 1

    # ps_spec = FLAGS.ps_hosts.split(",")
    # worker_spec = FLAGS.worker_hosts.split(",")
    # num_workers = len(worker_spec)
    # cluster = tf.train.ClusterSpec({
    #     "ps": ps_spec,
    #     "worker": worker_spec})
    # server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    # if FLAGS.job_name == "ps":
    #     with tf.device("/cpu:0"):
    #         server.join()
    #     return

    tf.logging.set_verbosity(tf.logging.DEBUG)
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

        # =================================================================== #
        # Create a dataset provider and batches.
        # =================================================================== #
        with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=FLAGS.num_readers,
                common_queue_capacity=120 * FLAGS.batch_size * num_gpus,
                common_queue_min=80 * FLAGS.batch_size * num_gpus,
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
        gclasses, glocalisations, gscores = \
            ron_net.bboxes_encode(glabels, gbboxes, ron_anchors, positive_threshold=FLAGS.match_threshold, ignore_threshold=FLAGS.neg_threshold)

        # each size of the batch elements
        # include one image, three others(gclasses, glocalisations, gscores)
        batch_shape = [1] + [len(ron_anchors)] * 3

        # Training batches and queue.
        r = tf.train.batch(
            tf_utils.reshape_list([image, gclasses, glocalisations, gscores]),
            batch_size=FLAGS.batch_size * num_gpus,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=120 * FLAGS.batch_size * num_gpus)
        all_batch = tf_utils.reshape_list(r, batch_shape)
        b_image = tf.split(all_batch[0], num_or_size_splits=num_gpus, axis=0)
        _b_gclasses = [tf.split(b, num_or_size_splits=num_gpus, axis=0) for b in all_batch[1]]
        b_gclasses = [_ for _ in zip(*_b_gclasses)]
        _b_glocalisations = [tf.split(b, num_or_size_splits=num_gpus, axis=0) for b in all_batch[2]]
        b_glocalisations = [_ for _ in zip(*_b_glocalisations)]
        _b_gscores = [tf.split(b, num_or_size_splits=num_gpus, axis=0) for b in all_batch[3]]
        b_gscores = [_ for _ in zip(*_b_gscores)]

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # =================================================================== #
    # Configure the optimization procedure.
    # =================================================================== #
    learning_rate = tf_utils.configure_learning_rate(FLAGS,
                                                     dataset.num_samples,
                                                     global_step)
    optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)
    summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    # Construct RON network.
    arg_scope = ron_net.arg_scope(weight_decay=FLAGS.weight_decay, data_format=DATA_FORMAT)

    reuse_variables = False
    tower_grads = []
    loss_list = []
    with slim.arg_scope(arg_scope):
        for index in range(num_gpus):
            with tf.device('/gpu:%d' % index):
                predictions, logits, objness_pred, objness_logits, localisations, end_points = ron_net.net(b_image[index], is_training=True, reuse = reuse_variables)
                # Add loss function.
                ron_net.losses(logits, localisations, objness_logits, objness_pred,
                               b_gclasses[index], b_glocalisations[index], b_gscores[index],
                               match_threshold = FLAGS.match_threshold,
                               neg_threshold = FLAGS.neg_threshold,
                               objness_threshold = FLAGS.objectness_thres,
                               negative_ratio=FLAGS.negative_ratio,
                               alpha=FLAGS.loss_alpha,
                               beta=FLAGS.loss_beta,
                               label_smoothing=FLAGS.label_smoothing)
                reuse_variables = True
                # and returns a train_tensor and summary_op
                loss = tf.losses.get_total_loss()
                loss_list.append(loss)
                # Variables to train.
                variables_to_train = tf_utils.get_variables_to_train(FLAGS)
                # Create gradient updates.
                grads = optimizer.compute_gradients(loss, variables_to_train)
                tower_grads.append(grads)

    reduce_grads = average_gradients(tower_grads)
    total_loss = tf.reduce_mean(tf.stack(loss_list, axis=0), axis=0)
    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss', total_loss))
    # =================================================================== #
    # Configure the moving averages.
    # =================================================================== #
    if FLAGS.moving_average_decay:
        moving_average_variables = slim.get_model_variables()
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, global_step)
    else:
        moving_average_variables, variable_averages = None, None

    if FLAGS.moving_average_decay:
        # Update ops executed locally by trainer.
        update_ops.append(variable_averages.apply(moving_average_variables))

    grad_updates = optimizer.apply_gradients(reduce_grads, global_step=global_step)
    update_ops.append(grad_updates)
    update_op = tf.group(*update_ops)
    train_tensor = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    # =================================================================== #
    # Kicks off the training.
    # =================================================================== #
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    saver = tf.train.Saver(max_to_keep=5,
                           keep_checkpoint_every_n_hours = FLAGS.save_interval_secs/3600.,
                           write_version=2,
                           pad_step_number=False)

    slim.learning.train(
        train_tensor,
        logdir=FLAGS.model_dir,
        master='',
        is_chief=True,
        init_fn=tf_utils.get_init_fn(FLAGS, os.path.join(FLAGS.data_dir, 'vgg_16.ckpt')),
        summary_op=summary_op,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        saver=saver,
        save_interval_secs=FLAGS.save_interval_secs,
        session_config=config,
        session_wrapper=None,
        sync_optimizer=None)


if __name__ == '__main__':
    tf.app.run()

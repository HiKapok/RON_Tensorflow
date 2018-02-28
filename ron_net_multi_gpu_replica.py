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
import time

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
    'num_readers', 6,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 6,
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
# =========================================================================== #
# Multi-GPU training Flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task performs the variable "
                     "initialization ")
tf.app.flags.DEFINE_integer("num_gpus", 1,
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
# since we use single GPU for each worker, so there is no need for average gradients
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

def build_model(n_workers, worker_index, is_chief, replicas_to_aggregate):
    with tf.device('/cpu:0'):
        global_step = slim.create_global_step()
        # Select the dataset.
        dataset = dataset_factory.get_replica_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.data_dir, num_workers = n_workers, worker_index = worker_index)

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
                common_queue_capacity=20 * FLAGS.batch_size,
                common_queue_min=10 * FLAGS.batch_size,
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
        image, glabels, gbboxes = \
            image_preprocessing_fn(image, glabels, gbboxes,
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
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=40 * FLAGS.batch_size)
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
        # and returns a train_tensor and summary_op
        total_loss = tf.losses.get_total_loss()
        # =================================================================== #
        # Configure the optimization procedure.
        # =================================================================== #
        decay_steps = int(dataset.num_samples / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)

        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')

        opt = tf.train.SyncReplicasOptimizer(
            tf.train.GradientDescentOptimizer(learning_rate), use_locking=False,
            replicas_to_aggregate=replicas_to_aggregate,
            total_num_replicas=n_workers)

        train_op = opt.minimize(total_loss, global_step=global_step)
        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        if is_chief:
            # Add summaries for losses and extra losses.
            for loss in tf.get_collection(tf.GraphKeys.LOSSES):
                summaries.add(tf.summary.scalar(loss.op.name, loss))
            for loss in tf.get_collection('EXTRA_LOSSES'):
                summaries.add(tf.summary.scalar(loss.op.name, loss))
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))
            summaries.add(tf.summary.scalar('total_loss', total_loss))

            if FLAGS.moving_average_decay:
                variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())
            else:
                variables_averages_op = tf.no_op()
            with tf.control_dependencies([variables_averages_op, train_op]):
                train_op = tf.no_op()
        return global_step, total_loss, train_op, opt, learning_rate, summaries

def create_done_queue(num_workers):
    with tf.device("/job:ps/task:0"):
        return tf.FIFOQueue(num_workers, tf.int32, shared_name="done_queue0")
# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    if not FLAGS.data_dir:
        raise ValueError('You must supply the dataset directory with --data_dir')
    # distribution check
    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")
    if FLAGS.task_index is None or FLAGS.task_index =="":
        raise ValueError("Must specify an explicit `task_index`")

    tf_logging.info("job name = %s" % FLAGS.job_name)
    tf_logging.info("task index = %d" % FLAGS.task_index)

    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    num_workers = len(worker_spec)
    cluster = tf.train.ClusterSpec({
        "ps": ps_spec,
        "worker": worker_spec})
    kill_ps_queue = create_done_queue(num_workers)
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        with tf.Session(server.target) as sess:
            for i in range(num_workers):
                sess.run(kill_ps_queue.dequeue())
        # with tf.device("/cpu:0"):
        #     server.join()
        return

    is_chief = (FLAGS.task_index == 0)

    if FLAGS.replicas_to_aggregate is None:
        replicas_to_aggregate = num_workers
    else:
        replicas_to_aggregate = FLAGS.replicas_to_aggregate

    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, ps_device="/job:ps", cluster=cluster)):
        with tf.device('/gpu:0'):
            global_step, total_loss, train_op, optimizer, learning_rate, summaries = build_model(num_workers, FLAGS.task_index, is_chief, replicas_to_aggregate)

        saver = tf.train.Saver(max_to_keep=5,
                               keep_checkpoint_every_n_hours = FLAGS.save_interval_secs/3600.,
                               write_version=2,
                               pad_step_number=False)
        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')
        #init_op = tf.global_variables_initializer()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())

        if is_chief:
            chief_queue_runner = optimizer.get_chief_queue_runner()
            token_nums = max(replicas_to_aggregate - num_workers, 0)
            init_tokens_op = optimizer.get_init_tokens_op(token_nums)

        kill_ps_enqueue_op = kill_ps_queue.enqueue(1)

        sv = tf.train.Supervisor(is_chief = is_chief,
                                logdir = FLAGS.model_dir,
                                init_op = init_op,
                                init_fn = tf_utils.get_init_fn(FLAGS, os.path.join(FLAGS.data_dir, 'vgg_16.ckpt')),
                                summary_op = summary_op,
                                recovery_wait_secs = 1,
                                saver = saver,
                                global_step = global_step,
                                save_model_secs = FLAGS.save_interval_secs,
                                save_summaries_secs = FLAGS.save_summaries_secs)

        if is_chief:
            tf_logging.info("Worker %d: Initializing session..." % FLAGS.task_index)
        else:
            tf_logging.info("Worker %d: Waiting for session to be initialized..." %
                  FLAGS.task_index)
        # Each worker only needs to contact the PS task(s) and the local worker task.
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)#, device_filters=['/job:ps', '/job:worker/task:%d' % FLAGS.task_index])

        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
        tf_logging.info("Worker %d: Session initialization complete." % FLAGS.task_index)

        if is_chief:
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_tokens_op)

        step = 0
        while not sv.should_stop():
            start_time = time.time()
            cur_global_step, cur_loss, cur_lr, _ = sess.run([global_step, total_loss, learning_rate, train_op])

            time_elapsed = time.time() - start_time
            if step % FLAGS.log_every_n_steps == 0:
                tf_logging.info('Current Speed: {:5.3f}sec/batch'.format(time_elapsed))
                tf_logging.info('Current Loss: {:5.3f}'.format(cur_loss))
                tf_logging.info('Global Step: %s', cur_global_step)
                tf_logging.info('Current Learning Rate: {}'.format(cur_lr))
            step += 1

        sess.run(kill_ps_enqueue_op)
        tf_logging.info('kill_ps_enqueue_op done....')
    sv.stop()


if __name__ == '__main__':
    tf.app.run()

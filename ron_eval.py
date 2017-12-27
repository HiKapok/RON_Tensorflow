import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.ops import control_flow_ops

import time
from datetime import datetime
import numpy as np

from scipy.misc import imread, imsave, imshow, imresize

from datasets import dataset_factory
from nets import nets_factory
from nets import ssd_common
from preprocessing import preprocessing_factory
import tf_utils

import tf_extended as tfe

import draw_toolbox


slim = tf.contrib.slim

# export CUDA_VISIBLE_DEVICES=''
DATA_FORMAT = 'NHWC' #'NCHW'

# =========================================================================== #
# General Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'test_dir', './eval_logs/',
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'num_cpu_threads', 3,
    'The number of cpu cores used to train.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.1, 'GPU memory fraction to use.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 1,
    'The frequency with which logs are print.')
# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', 'pascalvoc_2007', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', '../PASCAL/VOC2007TEST/TF/', 'The directory where the dataset files are stored.')
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

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_integer(
    'select_threshold', 0.4, 'Class-specific confidence score threshold for selecting a box.')
tf.app.flags.DEFINE_integer(
    'nms_threshold', 0.5, 'Matching threshold in NMS algorithm.')
tf.app.flags.DEFINE_integer(
    'nms_topk', 5, 'Number of total object to keep after NMS.')
# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'checkpoint_path', './model/model.ckpt-13417', #None, #'./checkpoints/ssd_300_vgg.ckpt',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', None,
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS

def flaten_predict(predictions, objness_pred, localisations):
    predictions_shape = tfe.get_shape(predictions[0], 5)
    batch_size = predictions_shape[0]
    num_classes = predictions_shape[-1]

    flaten_pred = []
    flaten_labels = []
    flaten_objness = []
    flaten_locations = []
    flaten_scores = []
    flaten_cutoff_scores = []

    for i in range(len(predictions)):
        flaten_labels.append(tf.reshape(tf.argmax(predictions[i], -1), [batch_size, -1]))
        flaten_pred.append(tf.reshape(predictions[i], [batch_size, -1, num_classes]))
        flaten_objness.append(tf.reshape(objness_pred[i], [batch_size, -1]))
        flaten_locations.append(tf.reshape(localisations[i], [batch_size, -1, 4]))
        flaten_scores.append(tf.expand_dims(flaten_objness[i], axis=-1) * flaten_pred[i])
        max_mask = tf.equal(tf.reduce_max(predictions[i], axis=-1, keep_dims=True), predictions[i])
        flaten_cutoff_scores.append(tf.cast(max_mask, predictions[i].dtype) * predictions[i])

    return flaten_scores, flaten_cutoff_scores, flaten_locations, flaten_labels


# =========================================================================== #
# Main eval routine.
# =========================================================================== #
def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():
        # Select the dataset.
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        # Get the RON network and its anchors.
        ron_class = nets_factory.get_network(FLAGS.model_name)
        ron_params = ron_class.default_params._replace(num_classes=FLAGS.num_classes)
        ron_net = ron_class(ron_params)
        ron_shape = ron_net.params.img_shape
        ron_anchors = ron_net.anchors(ron_shape)

        tf_utils.print_configuration(FLAGS.__flags, ron_params,
                                     dataset.data_sources, FLAGS.test_dir)
        # =================================================================== #
        # Create a dataset provider and batches.
        # =================================================================== #
        with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=FLAGS.num_readers,
                common_queue_capacity=20,
                common_queue_min=10,
                shuffle=True)
        # Get for RON network: image, labels, bboxes.
        # (ymin, xmin, ymax, xmax) fro gbboxes
        [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                         'object/label',
                                                         'object/bbox'])
        #### DEBUG ####
        #image = tf.Print(image, [shape, glabels, gbboxes], message='before preprocess: ', summarize=20)
        # Select the preprocessing function.
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training=False)

        # Pre-processing image, labels and bboxes.
        image, glabels, gbboxes, bbox_img = image_preprocessing_fn(image, glabels, gbboxes,
                                   out_shape=ron_shape,
                                   data_format=DATA_FORMAT)

        #### DEBUG ####
        #image = tf.Print(image, [shape, glabels, gbboxes], message='after preprocess: ', summarize=20)

        # Construct RON network.
        arg_scope = ron_net.arg_scope(data_format=DATA_FORMAT)
        with slim.arg_scope(arg_scope):
            predictions, _, objness_pred, _, localisations, _ = ron_net.net(tf.expand_dims(image, axis=0), is_training=False)
            localisations = ron_net.bboxes_decode(localisations, ron_anchors)
            flaten_scores, flaten_cutoff_scores, flaten_locations, _ = flaten_predict(predictions, objness_pred, localisations)
        # dict from class index to scores or bboxes those are selected depends on threshold
        # scores or bboxes in different layers are all concated together
        dict_scores, dict_bboxes = ssd_common.tf_ssd_bboxes_select(
                flaten_cutoff_scores, flaten_locations, select_threshold=FLAGS.select_threshold, num_classes=21, ignore_class=0)

        dict_bboxes = tfe.bboxes.bboxes_clip(bbox_img, dict_bboxes)
        dict_scores, dict_bboxes = tfe.bboxes.bboxes_sort(dict_scores, dict_bboxes,
                                                            top_k=FLAGS.nms_topk * 10)
        dict_scores, dict_bboxes = tfe.bboxes.bboxes_nms_batch(dict_scores, dict_bboxes, nms_threshold=FLAGS.nms_threshold, keep_top_k=FLAGS.nms_topk)
        # Resize bboxes to original image shape.
        dict_bboxes = tfe.bboxes.bboxes_resize(bbox_img, dict_bboxes)

        flaten_nms_scores = tf.concat(list(dict_scores.values()), axis=-1)
        flaten_nms_bboxes = tf.concat(list(dict_bboxes.values()), axis=-2)
        class_labels = [label for label in dict_scores.keys() for _ in range(FLAGS.nms_topk)]

        # configure model restore
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        tf.logging.info('Restoring model from %s. Ignoring missing vars: %s' % (checkpoint_path, FLAGS.ignore_missing_vars))

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
            variables_to_restore = variable_averages.variables_to_restore()
        else:
            variables_to_restore = slim.get_variables_to_restore()

        init_fn = slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore,
            ignore_missing_vars=FLAGS.ignore_missing_vars)

        def wrapper_debug(sess):
            sess = tf_debug.LocalCLIDebugWrapperSession(sess, thread_name_filter="MainThread$")
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            return sess

        # no need for specify local_variables_initializer and tables_initializer, Supervisor will do this via default local_init_op
        init_op = tf.group(tf.global_variables_initializer())
        # Pass the init function to the supervisor.
        # - The init function is called _after_ the variables have been initialized by running the init_op.
        # - manage summary in current process by ourselves for memory saving
        # - no need to specify global_step, supervisor will find this automately
        # - initialize order: checkpoint -> local_init_op -> init_op -> init_func
        sv = tf.train.Supervisor(logdir=FLAGS.test_dir, init_fn = init_fn, init_op = init_op, summary_op = None, save_model_secs=0)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction)
        config = tf.ConfigProto(log_device_placement = True, allow_soft_placement=True, intra_op_parallelism_threads = FLAGS.num_cpu_threads, inter_op_parallelism_threads = FLAGS.num_cpu_threads, gpu_options = gpu_options)

        cur_step = 0
        tf.logging.info(datetime.now().strftime('Evaluation Start: %Y-%m-%d %H:%M:%S'))

        with sv.managed_session(config=config) as sess:
            while True:
                if sv.should_stop():
                    tf.logging.info('Supervisor emited finish!')
                    break

                start_time = time.time()
                with tf.device('/gpu:0'):
                    image_input_ , _, _, scores_, bboxes_ = sess.run([image, glabels, gbboxes, flaten_nms_scores, flaten_nms_bboxes])
                    # print(image_input_)
                    print(scores_)
                    print(bboxes_)
                    img_to_draw = np.copy(preprocessing_factory.ssd_vgg_preprocessing.np_image_unwhitened(image_input_))
                    img_to_draw = draw_toolbox.bboxes_draw_on_img(img_to_draw, class_labels, scores_[0], bboxes_[0], thickness=2)
                    imsave('./Debug/{}.jpg'.format(cur_step), img_to_draw)
                time_elapsed = time.time() - start_time
                if cur_step % FLAGS.log_every_n_steps == 0:
                    tf.logging.info('Eval Speed: {:5.3f}sec/image'.format(time_elapsed))

                cur_step += 1

        tf.logging.info(datetime.now().strftime('Evaluation Finished: %Y-%m-%d %H:%M:%S'))

if __name__ == '__main__':
    tf.app.run()


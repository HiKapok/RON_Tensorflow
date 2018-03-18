import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.ops import control_flow_ops

import time
from datetime import datetime
import numpy as np
import pickle
import os

import xml.etree.ElementTree as ET

from scipy.misc import imread, imsave, imshow, imresize

from datasets import dataset_factory
from datasets import voc_eval
from datasets import pascalvoc_2007
from datasets import pascalvoc_common

from nets import nets_factory
from nets import ssd_common
from preprocessing import preprocessing_factory
import tf_utils

import tf_extended as tfe
from tf_extended import tensors as tfe_tensors

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
    'num_readers', 2,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 2,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'num_cpu_threads', 6,
    'The number of cpu cores used to train.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1., 'GPU memory fraction to use.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 5,
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
    'dataset_dir', '../PASCAL/tfrecords/VOC2007/TF_test/', 'The directory where the dataset files are stored.')
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
tf.app.flags.DEFINE_float(
    'select_threshold', 0.6, 'Class-specific confidence score threshold for selecting a box.')
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.4, 'Matching threshold in NMS algorithm.')
tf.app.flags.DEFINE_float(
    'objectness_thres', 0.95, 'threshold for the objectness to indicate the exist of object in that location.')
tf.app.flags.DEFINE_integer(
    'nms_topk_percls', 10, 'Number of object for each class to keep after NMS.')
tf.app.flags.DEFINE_integer(
    'nms_topk', 20, 'Number of total object to keep after NMS.')
# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'checkpoint_path', './model/model.ckpt-120055', #None, #'./checkpoints/ssd_300_vgg.ckpt',
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

    if batch_size > 1:
        raise ValueError('only batch_size 1 is supported.')

    flaten_pred = []
    flaten_labels = []
    flaten_objness = []
    flaten_locations = []
    flaten_scores = []

    for i in range(len(predictions)):
        flaten_pred.append(tf.reshape(predictions[i], [batch_size, -1, num_classes]))
        flaten_objness.append(tf.reshape(objness_pred[i], [batch_size, -1]))
        cls_pred = tf.expand_dims(flaten_objness[i], axis=-1) * flaten_pred[i]
        flaten_scores.append(tf.reshape(cls_pred, [batch_size, -1, num_classes]))
        #flaten_scores.append(tf.reshape(tf.reduce_max(cls_pred, -1), [batch_size, -1]))
        flaten_labels.append(tf.reshape(tf.argmax(cls_pred, -1), [batch_size, -1]))
        flaten_locations.append(tf.reshape(localisations[i], [batch_size, -1, 4]))
    # assume batch_size is always 1
    total_scores = tf.squeeze(tf.concat(flaten_scores, 1), 0)
    total_objness = tf.squeeze(tf.concat(flaten_objness, 1), 0)
    total_locations = tf.squeeze(tf.concat(flaten_locations, 1), 0)
    total_labels = tf.squeeze(tf.concat(flaten_labels, 1), 0)
    # remove bboxes that are not foreground
    non_background_mask = tf.greater(total_labels, 0)
    # remove bboxes that have scores lower than select_threshold
    #bbox_mask = tf.logical_and(non_background_mask, tf.greater(total_scores, FLAGS.select_threshold))
    # total_objness = tf.Print(total_objness, [total_objness])
    bbox_mask = tf.logical_and(non_background_mask, tf.greater(total_objness, FLAGS.objectness_thres))
    return tf.boolean_mask(total_scores, bbox_mask), tf.boolean_mask(total_labels, bbox_mask), tf.boolean_mask(total_locations, bbox_mask)

def tf_bboxes_nms(scores, labels, bboxes, nms_threshold = 0.5, keep_top_k = 200, mode = 'union', scope=None):
    with tf.name_scope(scope, 'tf_bboxes_nms', [scores, labels, bboxes]):
        # get the cls_score for the most-likely class
        scores = tf.reduce_max(scores, -1)
        # apply threshold
        bbox_mask = tf.greater(scores, FLAGS.select_threshold)
        scores, labels, bboxes = tf.boolean_mask(scores, bbox_mask), tf.boolean_mask(labels, bbox_mask), tf.boolean_mask(bboxes, bbox_mask)
        num_anchors = tf.shape(scores)[0]
        def nms_proc(scores, labels, bboxes):
            # sort all the bboxes
            scores, idxes = tf.nn.top_k(scores, k = num_anchors, sorted = True)
            labels, bboxes = tf.gather(labels, idxes), tf.gather(bboxes, idxes)

            ymin = bboxes[:, 0]
            xmin = bboxes[:, 1]
            ymax = bboxes[:, 2]
            xmax = bboxes[:, 3]

            vol_anchors = (xmax - xmin) * (ymax - ymin)

            nms_mask = tf.cast(tf.ones_like(scores, dtype=tf.int8), tf.bool)
            keep_mask = tf.cast(tf.zeros_like(scores, dtype=tf.int8), tf.bool)

            def safe_divide(numerator, denominator):
                return tf.where(tf.greater(denominator, 0), tf.divide(numerator, denominator), tf.zeros_like(denominator))

            def get_scores(bbox, nms_mask):
                # the inner square
                inner_ymin = tf.maximum(ymin, bbox[0])
                inner_xmin = tf.maximum(xmin, bbox[1])
                inner_ymax = tf.minimum(ymax, bbox[2])
                inner_xmax = tf.minimum(xmax, bbox[3])
                h = tf.maximum(inner_ymax - inner_ymin, 0.)
                w = tf.maximum(inner_xmax - inner_xmin, 0.)
                inner_vol = h * w
                this_vol = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if mode == 'union':
                    union_vol = vol_anchors - inner_vol  + this_vol
                elif mode == 'min':
                    union_vol = tf.minimum(vol_anchors, this_vol)
                else:
                    raise ValueError('unknown mode to use for nms.')
                return safe_divide(inner_vol, union_vol) * tf.cast(nms_mask, tf.float32)

            def condition(index, nms_mask, keep_mask):
                return tf.logical_and(tf.reduce_sum(tf.cast(nms_mask, tf.int32)) > 0, tf.less(index, keep_top_k))

            def body(index, nms_mask, keep_mask):
                # at least one True in nms_mask
                indices = tf.where(nms_mask)[0][0]
                bbox = bboxes[indices]
                this_mask = tf.one_hot(indices, num_anchors, on_value=False, off_value=True, dtype=tf.bool)
                keep_mask = tf.logical_or(keep_mask, tf.logical_not(this_mask))
                nms_mask = tf.logical_and(nms_mask, this_mask)

                nms_scores = get_scores(bbox, nms_mask)

                nms_mask = tf.logical_and(nms_mask, nms_scores < nms_threshold)
                return [index+1, nms_mask, keep_mask]

            index = 0
            [index, nms_mask, keep_mask] = tf.while_loop(condition, body, [index, nms_mask, keep_mask])
            return tf.boolean_mask(scores, keep_mask), tf.boolean_mask(labels, keep_mask), tf.boolean_mask(bboxes, keep_mask)

        return tf.cond(tf.less(num_anchors, 1), lambda: (scores, labels, bboxes), lambda: nms_proc(scores, labels, bboxes))

def tf_bboxes_nms_by_class(scores, labels, bboxes, nms_threshold = 0.5, keep_top_k = 200, mode = 'min', scope=None):
    with tf.name_scope(scope, 'tf_bboxes_nms_by_class', [scores, labels, bboxes]):
        num_anchors = tf.shape(scores)[0]
        def nms_proc(scores, labels, bboxes):
            # sort all the bboxes
            scores, idxes = tf.nn.top_k(scores, k = num_anchors, sorted = True)
            labels, bboxes = tf.gather(labels, idxes), tf.gather(bboxes, idxes)

            ymin = bboxes[:, 0]
            xmin = bboxes[:, 1]
            ymax = bboxes[:, 2]
            xmax = bboxes[:, 3]

            vol_anchors = (xmax - xmin) * (ymax - ymin)

            nms_mask = tf.cast(tf.ones_like(scores, dtype=tf.int8), tf.bool)
            nms_mask = tf.logical_and(nms_mask, tf.greater(scores, FLAGS.select_threshold))
            keep_mask = tf.cast(tf.zeros_like(scores, dtype=tf.int8), tf.bool)

            def safe_divide(numerator, denominator):
                return tf.where(tf.greater(denominator, 0), tf.divide(numerator, denominator), tf.zeros_like(denominator))

            def get_scores(bbox, nms_mask):
                # the inner square
                inner_ymin = tf.maximum(ymin, bbox[0])
                inner_xmin = tf.maximum(xmin, bbox[1])
                inner_ymax = tf.minimum(ymax, bbox[2])
                inner_xmax = tf.minimum(xmax, bbox[3])
                h = tf.maximum(inner_ymax - inner_ymin, 0.)
                w = tf.maximum(inner_xmax - inner_xmin, 0.)
                inner_vol = h * w
                this_vol = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if mode == 'union':
                    union_vol = vol_anchors - inner_vol  + this_vol
                elif mode == 'min':
                    union_vol = tf.minimum(vol_anchors, this_vol)
                else:
                    raise ValueError('unknown mode to use for nms.')
                return safe_divide(inner_vol, union_vol) * tf.cast(nms_mask, tf.float32)

            def condition(index, nms_mask, keep_mask):
                return tf.logical_and(tf.reduce_sum(tf.cast(nms_mask, tf.int32)) > 0, tf.less(index, keep_top_k))

            def body(index, nms_mask, keep_mask):
                # at least one True in nms_mask
                indices = tf.where(nms_mask)[0][0]
                bbox = bboxes[indices]
                this_mask = tf.one_hot(indices, num_anchors, on_value=False, off_value=True, dtype=tf.bool)

                this_keep_mask = tf.one_hot(idxes[indices], num_anchors, on_value=True, off_value=False, dtype=tf.bool)
                keep_mask = tf.logical_or(keep_mask, this_keep_mask)

                nms_mask = tf.logical_and(nms_mask, this_mask)

                nms_scores = get_scores(bbox, nms_mask)

                nms_mask = tf.logical_and(nms_mask, nms_scores < nms_threshold)
                return [index+1, nms_mask, keep_mask]

            index = 0
            [index, nms_mask, keep_mask] = tf.while_loop(condition, body, [index, nms_mask, keep_mask])
            return keep_mask
        def nms_by_cls_proc(scores, labels, bboxes):
            total_keep_mask = tf.map_fn(lambda _scores: nms_proc(_scores, labels, bboxes),
                                    tf.transpose(scores, perm=[1, 0]), parallel_iterations=10,
                                    back_prop=False,
                                    swap_memory=False,
                                    dtype=tf.bool,
                                    infer_shape=True)
            total_keep_mask = tf.transpose(total_keep_mask, perm=[1, 0])
            # scores in the keep places
            keep_scores = scores * tf.cast(total_keep_mask, scores.dtype)
            # get the max one in case one bbox is kept twice for different classes
            max_mask_scores = tf.reduce_max(keep_scores, -1)
            new_labels = tf.argmax(keep_scores, -1)
            # ignore bboxes those not been kept
            keep_mask = max_mask_scores > 0.
            return tf.boolean_mask(max_mask_scores, keep_mask), tf.boolean_mask(new_labels, keep_mask), tf.boolean_mask(bboxes, keep_mask)

        return tf.cond(tf.less(num_anchors, 1), lambda: (scores, labels, bboxes), lambda: nms_by_cls_proc(scores, labels, bboxes))

def tf_bboxes_nms_by_class_v1(scores, labels, bboxes, nms_threshold = 0.5, keep_top_k = 200, mode = 'min', scope=None):
    with tf.name_scope(scope, 'tf_bboxes_nms_by_class', [scores, labels, bboxes]):
        scores = tf.reduce_max(scores, -1)
        bbox_mask = tf.greater(scores, FLAGS.select_threshold)
        scores, labels, bboxes = tf.boolean_mask(scores, bbox_mask), tf.boolean_mask(labels, bbox_mask), tf.boolean_mask(bboxes, bbox_mask)
        num_anchors = tf.shape(scores)[0]
        def nms_proc(scores, labels, bboxes):
            # sort all the bboxes
            scores, idxes = tf.nn.top_k(scores, k = num_anchors, sorted = True)
            labels, bboxes = tf.gather(labels, idxes), tf.gather(bboxes, idxes)

            ymin = bboxes[:, 0]
            xmin = bboxes[:, 1]
            ymax = bboxes[:, 2]
            xmax = bboxes[:, 3]

            vol_anchors = (xmax - xmin) * (ymax - ymin)

            total_keep_mask = tf.cast(tf.zeros_like(scores, dtype=tf.int8), tf.bool)

            def safe_divide(numerator, denominator):
                return tf.where(tf.greater(denominator, 0), tf.divide(numerator, denominator), tf.zeros_like(denominator))

            def get_scores(bbox, nms_mask):
                # the inner square
                inner_ymin = tf.maximum(ymin, bbox[0])
                inner_xmin = tf.maximum(xmin, bbox[1])
                inner_ymax = tf.minimum(ymax, bbox[2])
                inner_xmax = tf.minimum(xmax, bbox[3])
                h = tf.maximum(inner_ymax - inner_ymin, 0.)
                w = tf.maximum(inner_xmax - inner_xmin, 0.)
                inner_vol = h * w
                this_vol = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if mode == 'union':
                    union_vol = vol_anchors - inner_vol  + this_vol
                elif mode == 'min':
                    union_vol = tf.minimum(vol_anchors, this_vol)
                else:
                    raise ValueError('unknown mode to use for nms.')
                return safe_divide(inner_vol, union_vol) * tf.cast(nms_mask, tf.float32)

            def condition(index, nms_mask, keep_mask):
                return tf.logical_and(tf.reduce_sum(tf.cast(nms_mask, tf.int32)) > 0, tf.less(index, keep_top_k))

            def body(index, nms_mask, keep_mask):
                # at least one True in nms_mask
                indices = tf.where(nms_mask)[0][0]
                bbox = bboxes[indices]
                this_mask = tf.one_hot(indices, num_anchors, on_value=False, off_value=True, dtype=tf.bool)
                keep_mask = tf.logical_or(keep_mask, tf.logical_not(this_mask))
                nms_mask = tf.logical_and(nms_mask, this_mask)

                nms_scores = get_scores(bbox, nms_mask)

                nms_mask = tf.logical_and(nms_mask, nms_scores < nms_threshold)
                return [index+1, nms_mask, keep_mask]
            def nms_loop_for_each(cls_index, total_keep_mask):
                index = 0
                nms_mask = tf.equal(tf.cast(cls_index, tf.int64), labels)
                keep_mask = tf.cast(tf.zeros_like(scores, dtype=tf.int8), tf.bool)

                [_, _, keep_mask] = tf.while_loop(condition, body, [index, nms_mask, keep_mask])
                total_keep_mask = tf.logical_or(total_keep_mask, keep_mask)

                return cls_index + 1, total_keep_mask
            cls_index = 1
            [_, total_keep_mask] = tf.while_loop(lambda cls_index, _: tf.less(cls_index, FLAGS.num_classes), nms_loop_for_each, [cls_index, total_keep_mask])
            indices_to_select = tf.where(total_keep_mask)
            select_mask = tf.cond(tf.less(tf.shape(indices_to_select)[0], keep_top_k + 1),
                                lambda: total_keep_mask,
                                lambda: tf.logical_and(total_keep_mask, tf.range(tf.cast(tf.shape(total_keep_mask)[0], tf.int64), dtype=tf.int64) < indices_to_select[keep_top_k][0]))
            return tf.boolean_mask(scores, select_mask), tf.boolean_mask(labels, select_mask), tf.boolean_mask(bboxes, select_mask)

        return tf.cond(tf.less(num_anchors, 1), lambda: (scores, labels, bboxes), lambda: nms_proc(scores, labels, bboxes))


def filter_boxes(scores, labels, bboxes, min_size_ratio, image_shape, net_input_shape):
    """Only keep boxes with both sides >= min_size and center within the image.
    min_size_ratio is the ratio relative to net input shape
    """
    # Scale min_size to match image scale
    min_size = tf.maximum(0.0001, min_size_ratio * tf.sqrt(tf.cast(image_shape[0] * image_shape[1], tf.float32) / (net_input_shape[0] * net_input_shape[1])))

    ymin = bboxes[:, 0]
    xmin = bboxes[:, 1]
    ymax = bboxes[:, 2]
    xmax = bboxes[:, 3]

    ws = xmax - xmin
    hs = ymax - ymin
    x_ctr = xmin + ws / 2.
    y_ctr = ymin + hs / 2.

    keep_mask = tf.logical_and(tf.greater(ws, min_size), tf.greater(hs, min_size))
    keep_mask = tf.logical_and(keep_mask, tf.greater(x_ctr, 0.))
    keep_mask = tf.logical_and(keep_mask, tf.greater(y_ctr, 0.))
    keep_mask = tf.logical_and(keep_mask, tf.less(x_ctr, 1.))
    keep_mask = tf.logical_and(keep_mask, tf.less(y_ctr, 1.))

    return tf.boolean_mask(scores, keep_mask), tf.boolean_mask(labels, keep_mask), tf.boolean_mask(bboxes, keep_mask)

def _process_image(directory, name):
    # Read the image file.
    #filename = os.path.join(directory, 'JPEGImages/' + name + '.jpg')
    filename = os.path.join(directory, 'JPEGImages/' + name + '.jpg')
    image_data = imread(filename, mode ='RGB')

    # Read the XML annotation file.
    filename = os.path.join(directory, 'Annotations/', name + '.xml')
    tree = ET.parse(filename)
    root = tree.getroot()

    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text)]
    # Find annotations.
    bboxes = []
    labels = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(pascalvoc_common.VOC_LABELS[label][0]))

        bbox = obj.find('bndbox')
        bboxes.append([float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ])
    return image_data, shape, labels, bboxes


# =========================================================================== #
# Main eval routine.
# =========================================================================== #
def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        # Get the RON network and its anchors.
        ron_class = nets_factory.get_network(FLAGS.model_name)
        ron_params = ron_class.default_params._replace(num_classes=FLAGS.num_classes)
        ron_net = ron_class(ron_params)
        ron_shape = ron_net.params.img_shape
        ron_anchors = ron_net.anchors(ron_shape)
        # Get for RON network: image, labels, bboxes.
        # (ymin, xmin, ymax, xmax) fro gbboxes

        image_input = tf.placeholder(tf.int32, shape=(None, None, 3))
        shape_input = tf.placeholder(tf.int32, shape=(2,))
        glabels_input = tf.placeholder(tf.int32, shape=(None,))
        gbboxes_input = tf.placeholder(tf.float32, shape=(None, 4))

        # Select the preprocessing function.
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training=False)

        # Pre-processing image, labels and bboxes.
        image, glabels, gbboxes, bbox_img = image_preprocessing_fn(image_input, glabels_input, gbboxes_input,
                                   out_shape=ron_shape,
                                   data_format=DATA_FORMAT)

        #### DEBUG ####
        #image = tf.Print(image, [shape, glabels, gbboxes], message='after preprocess: ', summarize=20)

        # Construct RON network.
        arg_scope = ron_net.arg_scope(is_training=False, data_format=DATA_FORMAT)
        with slim.arg_scope(arg_scope):
            predictions, _, objness_pred, _, localisations, _ = ron_net.net(tf.expand_dims(image, axis=0), is_training=False)
            bboxes = ron_net.bboxes_decode(localisations, ron_anchors)

            flaten_scores, flaten_labels, flaten_bboxes = flaten_predict(predictions, objness_pred, bboxes)
            #objness_pred = tf.reduce_max(tf.cast(tf.greater(objness_pred[-1], FLAGS.objectness_thres), tf.float32))

        flaten_bboxes = tfe.bboxes.bboxes_clip(bbox_img, flaten_bboxes)
        flaten_scores, flaten_labels, flaten_bboxes = filter_boxes(flaten_scores, flaten_labels, flaten_bboxes, 0.03, shape_input, [320., 320.])

        #flaten_scores, flaten_labels, flaten_bboxes = tf_bboxes_nms_by_class(flaten_scores, flaten_labels, flaten_bboxes, nms_threshold=FLAGS.nms_threshold, keep_top_k=FLAGS.nms_topk_percls, mode = 'union')
        flaten_scores, flaten_labels, flaten_bboxes = tf_bboxes_nms(flaten_scores, flaten_labels, flaten_bboxes, nms_threshold=FLAGS.nms_threshold, keep_top_k=FLAGS.nms_topk, mode = 'union')

        # Resize bboxes to original image shape.
        flaten_bboxes = tfe.bboxes.bboxes_resize(bbox_img, flaten_bboxes)

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
        config = tf.ConfigProto(log_device_placement = False, allow_soft_placement=True, intra_op_parallelism_threads = FLAGS.num_cpu_threads, inter_op_parallelism_threads = FLAGS.num_cpu_threads, gpu_options = gpu_options)

        cur_step = 0
        tf.logging.info(datetime.now().strftime('Evaluation Start: %Y-%m-%d %H:%M:%S'))

        detector_eval = voc_eval.DetectorEvalPascal('../PASCAL/VOC2007TEST/', './eval_logs/', set_type = 'test')
        num_images = pascalvoc_2007.SPLITS_TO_SIZES['test']

        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        all_boxes = [[[] for _ in range(num_images)] for _ in range(len(pascalvoc_common.VOC_CLASSES)+1)]
        output_dir = detector_eval.output_dir
        det_file = os.path.join(output_dir, 'detections.pkl')

        with sv.managed_session(config=config) as sess:
            while True:
                if sv.should_stop():
                    tf.logging.info('Supervisor emited finish!')
                    break
                if cur_step >= len(detector_eval.image_ids):
                    break
                start_time = time.time()

                input_datas = _process_image(detector_eval.image_ids[cur_step][0], detector_eval.image_ids[cur_step][1])
                with tf.device('/gpu:0'):
                    image_, shape_, _, _, scores_, labels_, bboxes_ = sess.run([image, shape_input, glabels, gbboxes, flaten_scores, flaten_labels, flaten_bboxes], feed_dict={image_input: input_datas[0],
                                    shape_input: input_datas[1],
                                    glabels_input: input_datas[2],
                                    gbboxes_input: input_datas[3]})
                    # print(image_)

                    # print(len(a),a[0].shape,a[1].shape,a[2].shape,a[3].shape)
                    # print(len(b),b[0].shape,b[1].shape,b[2].shape,b[3].shape)
                    # print(len(c),c[0].shape,c[1].shape,c[2].shape,c[3].shape)
                    print(scores_)
                    print(labels_)
                    print(bboxes_)
                    # print(a)
                    # print(FLAGS.objectness_thres)
                    img_to_draw = np.copy(preprocessing_factory.ssd_vgg_preprocessing.np_image_unwhitened(image_))
                    img_to_draw = draw_toolbox.bboxes_draw_on_img(img_to_draw, labels_, scores_, bboxes_, thickness=2)
                    imsave('./Debug/{}.jpg'.format(cur_step), img_to_draw)

                unique_labels = []
                for l in labels_:
                   if l not in unique_labels:
                      unique_labels.append(l)
                print('unique_labels:', unique_labels)
                # skip j = 0, because it's the background class
                for j in unique_labels:
                    mask = labels_ == j
                    boxes = bboxes_[mask]
                    # all detections are collected into:
                    #    all_boxes[cls][image] = N x 5 array of detections in
                    #    (x1, y1, x2, y2, score)
                    boxes[:, 0] *= shape_[0]
                    boxes[:, 2] *= shape_[0]
                    boxes[:, 1] *= shape_[1]
                    boxes[:, 3] *= shape_[1]

                    boxes[:,[0, 1]] = boxes[:,[1, 0]]
                    boxes[:,[2, 3]] = boxes[:,[3, 2]]
                    scores = scores_[mask]

                    cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                    print(cls_dets)
                    all_boxes[j][cur_step] = cls_dets

                time_elapsed = time.time() - start_time
                if cur_step % FLAGS.log_every_n_steps == 0:
                    tf.logging.info('Eval Speed: {:5.3f}sec/image, {}/{}'.format(time_elapsed, cur_step, len(detector_eval.image_ids)))

                cur_step += 1


        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        detector_eval.evaluate_detections(all_boxes)

        tf.logging.info(datetime.now().strftime('Evaluation Finished: %Y-%m-%d %H:%M:%S'))

if __name__ == '__main__':
    tf.app.run()

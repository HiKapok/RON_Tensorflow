# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Generic evaluation script that evaluates a SSD model
on a given dataset."""
import math
import sys
import six
import time

import numpy as np
import os
import tensorflow as tf
import tf_extended as tfe
import tf_utils
from tensorflow.python.framework import ops

import draw_toolbox

from scipy.misc import imread, imsave, imshow, imresize


from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

# =========================================================================== #
# Some default EVAL parameters
# =========================================================================== #
# List of recalls values at which precision is evaluated.
LIST_RECALLS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85,
                0.90, 0.95, 0.96, 0.97, 0.98, 0.99]
DATA_FORMAT = 'NHWC'

# =========================================================================== #
# SSD evaluation Flags.
# =========================================================================== #
# tf.app.flags.DEFINE_float(
#     'select_threshold', 0.75, 'Selection threshold.')
# tf.app.flags.DEFINE_float(
#     'objectness_thres', 0.95, 'threshold for the objectness to indicate the exist of object in that location.')
# tf.app.flags.DEFINE_integer(
#     'select_top_k', 100, 'Select top-k detected bounding boxes.')
# tf.app.flags.DEFINE_integer(
#     'keep_top_k', 10, 'Keep top-k detected objects.')
# tf.app.flags.DEFINE_float(
#     'nms_threshold', 0.4, 'Non-Maximum Selection threshold.')
# tf.app.flags.DEFINE_float(
#     'match_threshold', 0.5, 'Matching threshold with groundtruth objects.')

tf.app.flags.DEFINE_float(
    'select_threshold', 0.01, 'Selection threshold.')
tf.app.flags.DEFINE_float(
    'objectness_thres', 0.03, 'threshold for the objectness to indicate the exist of object in that location.')
tf.app.flags.DEFINE_integer(
    'select_top_k', 200, 'Select top-k detected bounding boxes.')
tf.app.flags.DEFINE_integer(
    'keep_top_k', 100, 'Keep top-k detected objects.')
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.4, 'Non-Maximum Selection threshold.')
tf.app.flags.DEFINE_float(
    'match_threshold', 0.5, 'Matching threshold with groundtruth objects.')

tf.app.flags.DEFINE_float(
    'loss_alpha', 1./3, 'Alpha parameter in the loss function.')
tf.app.flags.DEFINE_float(
    'loss_beta', 1./3, 'Beta parameter in the loss function.')
tf.app.flags.DEFINE_float(
    'negative_ratio', 3., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float(
    'neg_threshold', 0.3, 'Matching threshold for the negtive examples in the loss function.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

# =========================================================================== #
# Main evaluation flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')
tf.app.flags.DEFINE_float(
    'weight_decay', 0.0005, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')
tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', './model/model.ckpt-122044',#118815
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_string(
    'dataset_name', 'pascalvoc_2007', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', '../PASCAL/VOC_TF/VOC2007TEST_TF/', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'model_name', 'ron_320_vgg', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1., 'GPU memory fraction to use.')

FLAGS = tf.app.flags.FLAGS

def save_image_with_bbox(image, labels_, scores_, bboxes_):
    if not hasattr(save_image_with_bbox, "counter"):
        save_image_with_bbox.counter = 0  # it doesn't exist yet, so initialize it
    save_image_with_bbox.counter += 1

    img_to_draw = np.copy(image)#common_preprocessing.np_image_unwhitened(image))
    img_to_draw = draw_toolbox.bboxes_draw_on_img(img_to_draw, labels_, scores_, bboxes_, thickness=2)
    imsave(os.path.join('./Debug', '{}.jpg').format(save_image_with_bbox.counter), img_to_draw)
    return save_image_with_bbox.counter

def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        # Get the RON network and its anchors.
        ron_class = nets_factory.get_network(FLAGS.model_name)
        ron_params = ron_class.default_params._replace(num_classes=FLAGS.num_classes)
        ron_net = ron_class(ron_params)
        ron_shape = ron_net.params.img_shape
        ron_anchors = ron_net.anchors(ron_shape)

        # Select the preprocessing function.
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training=False)

        tf_utils.print_configuration(FLAGS.__flags, ron_params,
                                     dataset.data_sources, FLAGS.eval_dir)
        # =================================================================== #
        # Create a dataset provider and batches.
        # =================================================================== #
        with tf.device('/cpu:0'):
            with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
                provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    common_queue_capacity=2 * FLAGS.batch_size,
                    common_queue_min=FLAGS.batch_size,
                    shuffle=False)
            # Get for SSD network: image, labels, bboxes.
            [image_, shape, glabels, gbboxes, gdifficults] = provider.get(['image', 'shape',
                                                         'object/label',
                                                         'object/bbox',
                                                         'object/difficult'])

            # Pre-processing image, labels and bboxes.
            image, glabels, gbboxes, gbbox_img = \
                image_preprocessing_fn(image_, glabels, gbboxes,
                                       out_shape=ron_shape,
                                       data_format=DATA_FORMAT,
                                       difficults=None)

            # Encode groundtruth labels and bboxes.
            gclasses, glocalisations, gscores = \
                ron_net.bboxes_encode(glabels, gbboxes, ron_anchors)
            batch_shape = [1] * 5 + [len(ron_anchors)] * 3

            # Evaluation batch.
            r = tf.train.batch(
                tf_utils.reshape_list([image, glabels, gbboxes, gdifficults, gbbox_img,
                                       gclasses, glocalisations, gscores]),
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size,
                dynamic_pad=True)
            (b_image, b_glabels, b_gbboxes, b_gdifficults, b_gbbox_img, b_gclasses,
             b_glocalisations, b_gscores) = tf_utils.reshape_list(r, batch_shape)

        # =================================================================== #
        # SSD Network + Ouputs decoding.
        # =================================================================== #
        dict_metrics = {}
        arg_scope = ron_net.arg_scope(weight_decay=FLAGS.weight_decay,
                                        is_training=False,
                                        data_format=DATA_FORMAT)

        with slim.arg_scope(arg_scope):
            predictions, logits, objness_pred, objness_logits, localisations, end_points = \
                ron_net.net(b_image, is_training=False)
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

        variables_to_restore = slim.get_variables_to_restore()
        # Performing post-processing on CPU: loop-intensive, usually more efficient.
        with tf.device('/device:CPU:0'):
            # Detected objects from SSD output.
            localisations = ron_net.bboxes_decode(localisations, ron_anchors)
            filtered_predictions = []
            for i, objness in enumerate(objness_pred):
                filtered_predictions.append(tf.cast(tf.greater(objness, FLAGS.objectness_thres), tf.float32) * predictions[i])
            rscores, rbboxes = \
                ron_net.detected_bboxes(filtered_predictions, localisations,
                                        select_threshold=FLAGS.select_threshold,
                                        nms_threshold=FLAGS.nms_threshold,
                                        clipping_bbox=[0., 0., 1., 1.],
                                        top_k=FLAGS.select_top_k,
                                        keep_top_k=FLAGS.keep_top_k)
            labels_list = []
            for k, v in rscores.items():
                labels_list.append(tf.ones_like(v, tf.int32) * k)
            save_image_op = tf.py_func(save_image_with_bbox,
                                        [tf.cast(tf.squeeze(b_image, 0), tf.float32),
                                        tf.squeeze(tf.concat(labels_list, axis=1), 0),
                                        #tf.convert_to_tensor(list(rscores.keys()), dtype=tf.int64),
                                        tf.squeeze(tf.concat(list(rscores.values()), axis=1), 0),
                                        tf.squeeze(tf.concat(list(rbboxes.values()), axis=1), 0)],
                                        tf.int64, stateful=True)
            with tf.control_dependencies([save_image_op]):
                # Compute TP and FP statistics.
                num_gbboxes, tp, fp, rscores = \
                    tfe.bboxes_matching_batch(rscores.keys(), rscores, rbboxes,
                                              b_glabels, b_gbboxes, b_gdifficults,
                                              matching_threshold=0.5)



        # =================================================================== #
        # Evaluation metrics.
        # =================================================================== #
        with tf.device('/device:CPU:0'):
            dict_metrics = {}
            # First add all losses.
            for loss in tf.get_collection(tf.GraphKeys.LOSSES):
                dict_metrics[loss.op.name] = slim.metrics.streaming_mean(loss)
            # Extra losses as well.
            for loss in tf.get_collection('EXTRA_LOSSES'):
                dict_metrics[loss.op.name] = slim.metrics.streaming_mean(loss)

            # Add metrics to summaries and Print on screen.
            for name, metric in dict_metrics.items():
                # summary_name = 'eval/%s' % name
                summary_name = name
                op = tf.summary.scalar(summary_name, metric[0], collections=[])
                # op = tf.Print(op, [metric[0]], summary_name)
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

            # FP and TP metrics.
            tp_fp_metric = tfe.streaming_tp_fp_arrays(num_gbboxes, tp, fp, rscores)

            metrics_name = ('nobjects', 'ndetections', 'tp', 'fp', 'scores')
            for c in tp_fp_metric[0].keys():
                for _ in range(len(tp_fp_metric[0][c])):
                    dict_metrics['tp_fp_%s_%s' % (c, metrics_name[_])] = (tp_fp_metric[0][c][_],
                                                    tp_fp_metric[1][c][_])

            # for c in tp_fp_metric[0].keys():
            #     dict_metrics['tp_fp_%s' % c] = (tp_fp_metric[0][c],
            #                                     tp_fp_metric[1][c])

            # Add to summaries precision/recall values.
            aps_voc07 = {}
            aps_voc12 = {}
            for c in tp_fp_metric[0].keys():
                # Precison and recall values.
                prec, rec = tfe.precision_recall(*tp_fp_metric[0][c])

                # Average precision VOC07.
                v = tfe.average_precision_voc07(prec, rec)
                summary_name = 'AP_VOC07/%s' % c
                op = tf.summary.scalar(summary_name, v, collections=[])
                # op = tf.Print(op, [v], summary_name)
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
                aps_voc07[c] = v

                # Average precision VOC12.
                v = tfe.average_precision_voc12(prec, rec)
                summary_name = 'AP_VOC12/%s' % c
                op = tf.summary.scalar(summary_name, v, collections=[])
                # op = tf.Print(op, [v], summary_name)
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
                aps_voc12[c] = v

            # Mean average precision VOC07.
            summary_name = 'AP_VOC07/mAP'
            mAP = tf.add_n(list(aps_voc07.values())) / len(aps_voc07)
            op = tf.summary.scalar(summary_name, mAP, collections=[])
            op = tf.Print(op, [mAP], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

            # Mean average precision VOC12.
            summary_name = 'AP_VOC12/mAP'
            mAP = tf.add_n(list(aps_voc12.values())) / len(aps_voc12)
            op = tf.summary.scalar(summary_name, mAP, collections=[])
            op = tf.Print(op, [mAP], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # for i, v in enumerate(l_precisions):
        #     summary_name = 'eval/precision_at_recall_%.2f' % LIST_RECALLS[i]
        #     op = tf.summary.scalar(summary_name, v, collections=[])
        #     op = tf.Print(op, [v], summary_name)
        #     tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)


        # Split into values and updates ops.
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(dict_metrics)

        # =================================================================== #
        # Evaluation loop.
        # =================================================================== #
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        # Number of batches...
        num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
                checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path
        tf.logging.info('Evaluating %s' % checkpoint_path)

        # Standard evaluation loop.
        start = time.time()
        slim.evaluation.evaluate_once(
            master=FLAGS.master,
            checkpoint_path=checkpoint_path,
            logdir=FLAGS.eval_dir,
            num_evals=num_batches,
            eval_op=list(names_to_updates.values()),
            variables_to_restore=variables_to_restore,
            session_config=config)
        # Log time spent.
        elapsed = time.time()
        elapsed = elapsed - start
        print('Time spent : %.3f seconds.' % elapsed)
        print('Time spent per BATCH: %.3f seconds.' % (elapsed / num_batches))


if __name__ == '__main__':
    tf.app.run()

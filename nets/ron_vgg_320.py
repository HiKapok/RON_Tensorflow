# Copyright 2016 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Definition of 300 VGG-based RON network.

This model was initially introduced in:
RON: Single Shot MultiBox Detector
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg
https://arxiv.org/abs/1512.02325

Two variants of the model are defined: the 300x300 and 512x512 models, the
latter obtaining a slightly better accuracy on Pascal VOC.

Usage:
    with slim.arg_scope(ssd_vgg.ssd_vgg()):
        outputs, end_points = ssd_vgg.ssd_vgg(inputs)

This network port of the original Caffe model. The padding in TF and Caffe
is slightly different, and can lead to severe accuracy drop if not taken care
in a correct way!

In Caffe, the output size of convolution and pooling layers are computing as
following: h_o = (h_i + 2 * pad_h - kernel_h) / stride_h + 1

Nevertheless, there is a subtle difference between both for stride > 1. In
the case of convolution:
    top_size = floor((bottom_size + 2*pad - kernel_size) / stride) + 1
whereas for pooling:
    top_size = ceil((bottom_size + 2*pad - kernel_size) / stride) + 1
Hence implicitely allowing some additional padding even if pad = 0. This
behaviour explains why pooling with stride and kernel of size 2 are behaving
the same way in TensorFlow and Caffe.

Nevertheless, this is not the case anymore for other kernel sizes, hence
motivating the use of special padding layer for controlling these side-effects.

@@ssd_vgg_300
"""
import math
from collections import namedtuple

import numpy as np
import tensorflow as tf


from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

import tf_extended as tfe
from nets import custom_layers
from nets import ssd_common

slim = tf.contrib.slim


# =========================================================================== #
# RON class definition.
# =========================================================================== #
RONParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'prior_scaling'
                                         ])


class RONNet(object):
    """Implementation of the RON VGG-based 320 network.

    The default features layers with 320x320 image input are:
      conv4 ==> 40 x 40
      conv5 ==> 20 x 20
      conv6 ==> 10 x 10
      conv7 ==> 5 x 5

    The default image size used to train this network is 320x320.
    """
    default_params = RONParams(
        img_shape=(320, 320),
        num_classes=21,
        no_annotation_label=21,
        feat_layers=['block7','block6', 'block5', 'block4'],
        feat_shapes=[(5, 5), (10, 10), (20, 20), (40, 40)],
        anchor_sizes=[(224., 256.),
                      (160., 192.),
                      (96., 128.),
                      (32., 64.)],

        anchor_ratios=[[1, 2, 3, 1./2, 1./3],
                       [1, 2, 3, 1./2, 1./3],
                       [1, 2, 3, 1./2, 1./3],
                       [1, 2, 3, 1./2, 1./3]],
        anchor_steps=[64, 32, 16, 8],

        # feat_layers=['block7'],
        # feat_shapes=[(5, 5)],
        # anchor_sizes=[(224., 256.)],

        # anchor_ratios=[[1, 2, 3, 1./2, 1./3]],
        # anchor_steps=[64],

        anchor_offset=0.5,
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

    def __init__(self, params=None):
        """Init the RON net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, RONParams):
            self.params = params
        else:
            self.params = RONNet.default_params

    # ======================================================================= #
    def net(self, inputs,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ron_320_vgg'):
        """RON network definition.
        """
        r = ron_net(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        return r

    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        """Network arg_scope.
        """
        return ron_arg_scope(weight_decay, data_format=data_format)

    # ======================================================================= #
    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return ron_anchors_all_layers(img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)

    def bboxes_encode(self, labels, bboxes, anchors, positive_threshold=0.5, ignore_threshold=0.3,
                      scope=None):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            self.params.no_annotation_label,
            positive_threshold = positive_threshold,
            ignore_threshold = ignore_threshold,
            prior_scaling = self.params.prior_scaling,
            scope=scope)

    def bboxes_decode(self, feat_localizations, anchors,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def detected_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
        """Get the detected bounding boxes from the RON network output.
        """
        # Select top_k bboxes from predictions, and clip
        rscores, rbboxes = \
            ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            num_classes=self.params.num_classes)
        rscores, rbboxes = \
            tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)
        if clipping_bbox is not None:
            rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes

    def losses(self, logits, localisations, objness_logits, objness_pred,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               neg_threshold = 0.3,
               objness_threshold = 0.03,
               negative_ratio=3.,
               alpha=1./3,
               beta=1./3,
               label_smoothing=0.,
               scope='ron_losses'):
        """Define the RON network losses.
        """
        return ron_losses(logits, localisations, objness_logits, objness_pred,
                          gclasses, glocalisations, gscores,
                          match_threshold = match_threshold,
                          neg_threshold = neg_threshold,
                          objness_threshold = objness_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          beta=beta,
                          label_smoothing=label_smoothing,
                          scope=scope)


# =========================================================================== #
# RON tools...
# =========================================================================== #
def ron_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer RON default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of RON for the order.
    num_anchors = len(sizes) * len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)

    for i, r in enumerate(ratios):
        for j, s in enumerate(sizes):
            h[i*len(sizes) + j] = s / img_shape[0] / math.sqrt(r)
            w[i*len(sizes) + j] = s / img_shape[1] * math.sqrt(r)

    return y, x, h, w


def ron_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """

    #img_shape = tf.Print(img_shape, [layers_shape, anchor_sizes, anchor_ratios, anchor_steps], message='anchors: ', summarize=20)
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ron_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


# =========================================================================== #
# Functional definition of VGG-based RON 300.
# =========================================================================== #
def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def pred_cls_module(net_input, var_scope, num_anchors, num_classes):
  with tf.variable_scope(var_scope + '_inception1'):
    with tf.variable_scope('Branch_0'):
      branch_0 = slim.conv2d(net_input, 512, [3, 3], normalizer_fn = None, activation_fn = None, scope='Conv2d_3x3')
    with tf.variable_scope('Branch_1'):
      branch_1 = slim.conv2d(net_input, 512, [1, 1], normalizer_fn = None, activation_fn = None, scope='Conv2d_1x1')

    net_input = array_ops.concat([branch_0, branch_1], 3)
    # only activation after concat
    net_input = slim.batch_norm(net_input, activation_fn=tf.nn.relu)

  with tf.variable_scope(var_scope + '_inception2'):
    with tf.variable_scope('Branch_0'):
      branch_0 = slim.conv2d(net_input, 512, [3, 3], normalizer_fn = None, activation_fn = None, scope='Conv2d_3x3')
    with tf.variable_scope('Branch_1'):
      branch_1 = slim.conv2d(net_input, 512, [1, 1], normalizer_fn = None, activation_fn = None, scope='Conv2d_1x1')

    net_input = array_ops.concat([branch_0, branch_1], 3)
    # only activation after concat
    net_input = slim.batch_norm(net_input, activation_fn=tf.nn.relu)

    cls_pred = slim.conv2d(net_input, num_anchors * num_classes, [3, 3], activation_fn=None, scope='Conv2d_pred_3x3')

  cls_pred = custom_layers.channel_to_last(cls_pred)
  cls_pred = tf.reshape(cls_pred, tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])

  return cls_pred

def reg_bbox_module(net_input, var_scope, num_anchors):# = 'reg_bbox_@4'
  with tf.variable_scope(var_scope):
    net_input = slim.conv2d(net_input, 512, [3, 3], normalizer_fn = slim.batch_norm, scope='Conv2d_0_3x3')

    loc_pred = slim.conv2d(net_input, 4 * num_anchors, [3, 3], activation_fn=None, scope='Conv2d_1_3x3')

    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred, tensor_shape(loc_pred, 4)[:-1] + [num_anchors, 4])

  return loc_pred

# it seem's that no matter how many channals ref_map has, 512 will be used after deconv
def reverse_connection_module_with_pred(left_input, right_input, num_classes, num_anchors, var_scope):
  if right_input is None:
      ref_map = slim.conv2d(left_input, 512, [2, 2], stride=2, normalizer_fn = slim.batch_norm, scope = var_scope + '_conv_left')
  else:
      left_conv = slim.conv2d(left_input, 512, [3, 3], normalizer_fn = slim.batch_norm, scope = var_scope + '_conv_left')
      # remove BN for deconv, but leave Relu
      upsampling = slim.conv2d_transpose(right_input, 512, [2, 2], stride=2, normalizer_fn = None, scope = var_scope + '_deconv_right')
      ref_map = tf.nn.relu(left_conv + upsampling)

  objness_ref_map = slim.conv2d(ref_map, 512, [3, 3], normalizer_fn = slim.batch_norm, scope= var_scope + '_objectness')
  objectness_logits = tf.reshape(slim.conv2d(objness_ref_map, 2 * num_anchors, [3, 3], activation_fn = None, scope= var_scope + '_objectness_score'), tensor_shape(objness_ref_map, 4)[:-1]+[num_anchors, 2])

  # objectness_logits = tf.reshape(slim.conv2d(ref_map, 2 * num_anchors, [3, 3], activation_fn = None, scope= var_scope + '_objectness'), tensor_shape(ref_map, 4)[:-1]+[num_anchors, 2])

  return ref_map, objectness_logits, pred_cls_module(ref_map, var_scope, num_anchors, num_classes), reg_bbox_module(ref_map, var_scope, num_anchors)


def ron_net(inputs,
            num_classes=RONNet.default_params.num_classes,
            feat_layers=RONNet.default_params.feat_layers,
            anchor_sizes=RONNet.default_params.anchor_sizes,
            anchor_ratios=RONNet.default_params.anchor_ratios,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ron_320_vgg'):
    """RON net definition.
    """
    # if data_format == 'NCHW':
    #     inputs = tf.transpose(inputs, perm=(0, 3, 1, 2))

    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.variable_scope(scope, 'ron_320_vgg', [inputs], reuse=reuse):

        # Original VGG-16 blocks.
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['block1'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['block3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        end_points['block4'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['block5'] = net
        # different betweent SSD here
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        # Additional RON blocks.
        # Block 6
        net = slim.conv2d(net, 4096, [7, 7], scope='conv6')
        end_points['block6'] = net
        # Block 7: 1x1 conv, no padding.
        net = slim.conv2d(net, 4096, [1, 1], scope='conv7')
        end_points['block7'] = net

        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        objness_pred = []
        objness_logits = []

        # last_refmap = slim.conv2d(net, 512, [3, 3], scope='conv7_refmap')
        # end_points['block7_refmap'] = last_refmap
        cur_ref_map = None
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope('reverse_module'):
                #print(tfe.get_shape(end_points[layer], 4))
                cur_ref_map, objness, cls_pred, bbox_reg = reverse_connection_module_with_pred(end_points[layer], cur_ref_map, num_classes,\
                                              len(anchor_sizes[i]) * len(anchor_ratios[i]), var_scope = layer + '_reverse')
                predictions.append(prediction_fn(cls_pred))
                logits.append(cls_pred)
                obj_pred_neg_pos = prediction_fn(objness)
                #objness_pred.append(tf.ones_like(tf.slice(obj_pred_neg_pos, [0, 0,0,0,1], [-1, -1,-1,-1,1])))
                objness_pred.append(tf.slice(obj_pred_neg_pos, [0, 0, 0, 0, 1], [-1, -1, -1, -1, 1]))
                objness_logits.append(objness)
                localisations.append(bbox_reg)

        return predictions, logits, objness_pred, objness_logits, localisations, end_points

ron_net.default_image_size = 320


def ron_arg_scope(weight_decay=0.0005, data_format='NHWC'):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    # with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
    #                     activation_fn=tf.nn.relu,
    #                     weights_regularizer=slim.l2_regularizer(weight_decay),
    #                     weights_initializer=tf.contrib.layers.xavier_initializer(),
    #                     biases_initializer=tf.zeros_initializer()):
    #     with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.max_pool2d],
    #                         padding='SAME',
    #                         data_format=data_format):
    #         with slim.arg_scope([slim.batch_norm],
    #                         activation_fn=tf.nn.relu,
    #                         decay=0.997,
    #                         epsilon=1e-5,
    #                         scale=True,
    #                         data_format=data_format):
    #             with slim.arg_scope([custom_layers.pad2d,
    #                                  custom_layers.l2_normalization,
    #                                  custom_layers.channel_to_last],
    #                                 data_format=data_format) as sc:
    #                 return sc
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([slim.batch_norm],
                            # default no activation_fn for BN
                            activation_fn=None,
                            decay=0.997,
                            epsilon=1e-5,
                            scale=True,
                            fused=True,
                            data_format=data_format):
                with slim.arg_scope([custom_layers.pad2d,
                                     custom_layers.l2_normalization,
                                     custom_layers.channel_to_last],
                                    data_format=data_format) as sc:
                    return sc


# =========================================================================== #
# RON loss function.
# =========================================================================== #
def ron_losses(logits, localisations, objness_logits, objness_pred,
               gclasses, glocalisations, gscores,
               match_threshold = 0.5,
               neg_threshold = 0.3,
               objness_threshold = 0.03,
               negative_ratio=3.,
               alpha=1./3,
               beta=1./3,
               label_smoothing=0.,
               device='/cpu:0',
               scope=None):
    with tf.name_scope(scope, 'ron_losses'):
        # why rank 5, batch, height, width, num_anchors, num_classes
        logits_shape = tfe.get_shape(logits[0], 5)
        num_classes = logits_shape[-1]
        batch_size = logits_shape[0]

        # Flatten out all vectors
        flogits = []
        fobjness_logits = []
        fobjness_pred = []
        fgclasses = []
        fgscores = []
        flocalisations = []
        fglocalisations = []
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fobjness_logits.append(tf.reshape(objness_logits[i], [-1, 2]))
            fobjness_pred.append(tf.reshape(objness_pred[i], [-1]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
        # concat along different feature map (from last to front: layer7->layer4)
        logits = tf.concat(flogits, axis=0)
        objness_logits = tf.concat(fobjness_logits, axis=0)
        objness_pred = tf.concat(fobjness_pred, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        dtype = logits.dtype

        #num_nonzero = tf.count_nonzero(gclasses)
        #gclasses = tf.Print(gclasses, [num_nonzero], message='gscores non_zeros: ', summarize=20)
        # gscores = tf.Print(gscores, [gscores], message='gscores: ', summarize=50)

        # raw mask for positive > 0.5, and for negetive < 0.3
        # each positive examples has one label
        positive_mask = gclasses > 0
        fpositive_mask = tf.cast(positive_mask, dtype)
        n_positives = tf.reduce_sum(fpositive_mask)
        # negtive examples are those max_overlap is still lower than neg_threshold, note that some positive may also has lower jaccard

        #negtive_mask = tf.cast(tf.logical_not(positive_mask), dtype) * gscores < neg_threshold
        negtive_mask = tf.logical_and(tf.logical_not(positive_mask), gscores < neg_threshold)
        #negtive_mask = tf.logical_and(gscores < neg_threshold, tf.logical_not(positive_mask))
        fnegtive_mask = tf.cast(negtive_mask, dtype)
        n_negtives = tf.reduce_sum(fnegtive_mask)

        # random select hard negtive for objectness
        n_neg_to_select = tf.cast(negative_ratio * n_positives, tf.int32)
        n_neg_to_select = tf.minimum(n_neg_to_select, tf.cast(n_negtives, tf.int32))

        rand_neg_mask = tf.random_uniform(tfe.get_shape(gscores, 1), minval=0, maxval=1.) < tfe.safe_divide(tf.cast(n_neg_to_select, dtype), n_negtives, name='rand_select_objness')
        # include both random_select negtive and all positive examples
        final_neg_mask_objness = tf.stop_gradient(tf.logical_or(tf.logical_and(negtive_mask, rand_neg_mask), positive_mask))
        total_examples_for_objness = tf.reduce_sum(tf.cast(final_neg_mask_objness, dtype))
        # the label for objectness is all the positive
        objness_pred_label = tf.stop_gradient(tf.cast(positive_mask, tf.int32))

        # objness_pred = tf.Print(objness_pred, [objness_pred], message='objness_pred: ', summarize=50)

        # objectness score in all positive positions
        objness_pred_in_positive = tf.cast(positive_mask, dtype) * objness_pred
        # max objectness score in all positive positions
        max_objness_in_positive = tf.reduce_max(objness_pred_in_positive)
        # the position of max objectness score in all positive positions
        max_objness_mask = tf.equal(objness_pred_in_positive, max_objness_in_positive)


        # objectness mask for select real positive for detection
        objectness_mask = objness_pred > objness_threshold
        # positive for detection, and insure there is more than one positive to predict
        #cls_positive_mask = tf.stop_gradient(tf.logical_or(tf.logical_and(positive_mask, objectness_mask), max_objness_mask))
        cls_positive_mask = tf.stop_gradient(tf.logical_and(positive_mask, objectness_mask))
        cls_negtive_mask = tf.logical_and(objectness_mask, negtive_mask)
        #cls_negtive_mask = tf.logical_and(objectness_mask, tf.logical_not(cls_positive_mask))

        n_cls_negtives = tf.reduce_sum(tf.cast(cls_negtive_mask, dtype))

        fcls_positive_mask = tf.cast(cls_positive_mask, dtype)
        n_cls_positives = tf.reduce_sum(fcls_positive_mask)
        n_cls_neg_to_select = tf.cast(negative_ratio * n_cls_positives, tf.int32)
        n_cls_neg_to_select = tf.minimum(n_cls_neg_to_select, tf.cast(n_cls_negtives, tf.int32))
        # random selected negtive mask
        rand_cls_neg_mask = tf.random_uniform(tfe.get_shape(gscores, 1), minval=0, maxval=1.) < tfe.safe_divide(tf.cast(n_cls_neg_to_select, dtype), n_cls_negtives, name='rand_select_cls')
        # include both random_select negtive and all positive(positive is filtered by objectness)
        final_cls_neg_maks_objness = tf.stop_gradient(tf.logical_or(tf.logical_and(cls_negtive_mask, rand_cls_neg_mask), cls_positive_mask))
        total_examples_for_cls = tf.reduce_sum(tf.cast(final_cls_neg_maks_objness, dtype))

        # n_cls_neg_to_select = tf.Print(n_cls_neg_to_select, [n_cls_neg_to_select], message='n_cls_neg_to_select: ', summarize=20)
        # n_cls_positives = tf.Print(n_cls_positives, [n_cls_positives], message='n_cls_positives: ', summarize=20)
        # n_neg_to_select = tf.Print(n_neg_to_select, [n_neg_to_select], message='n_neg_to_select: ', summarize=20)
        # n_positives = tf.Print(n_positives, [n_positives], message='n_positives: ', summarize=20)

        # Add cross-entropy loss.
        with tf.name_scope('cross_entropy_pos'):
            #weights = (1. - alpha - beta) * tf.cast(final_cls_neg_maks_objness, dtype)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.stop_gradient(gclasses))

            loss = tf.cond(n_positives > 0., lambda: (1. - alpha - beta) * tf.reduce_mean(tf.boolean_mask(loss, final_cls_neg_maks_objness)), lambda: 0.)
            #loss = tf.reduce_mean(loss * weights)
            #loss = tf.reduce_sum(loss * weights)
            #loss = tfe.safe_divide(tf.reduce_sum(loss * weights), total_examples_for_cls, name='cls_loss')
            tf.losses.add_loss(loss)

        with tf.name_scope('cross_entropy_objectness'):
            #weights = alpha * tf.cast(final_neg_mask_objness, dtype)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=objness_logits, labels=objness_pred_label)
            loss = tf.cond(n_positives > 0., lambda: alpha * tf.reduce_mean(tf.boolean_mask(loss, final_neg_mask_objness)), lambda: 0.)
            #loss = tf.reduce_mean(loss * weights)
            #loss = tf.reduce_sum(loss * weights)
            #loss = tfe.safe_divide(tf.reduce_sum(loss * weights), total_examples_for_objness, name='objness_loss')
            tf.losses.add_loss(loss)

        # Add localization loss
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            #weights = tf.expand_dims(beta * tf.cast(fcls_positive_mask, dtype), axis=-1)
            loss = custom_layers.modified_smooth_l1(localisations, tf.stop_gradient(glocalisations), sigma = 3.)
            #loss = custom_layers.abs_smooth(localisations - tf.stop_gradient(glocalisations))

            loss = tf.cond(n_positives > 0., lambda: beta * n_positives / total_examples_for_objness * tf.reduce_mean(tf.boolean_mask(tf.reduce_sum(loss, axis=-1), tf.stop_gradient(positive_mask))), lambda: 0.)
            #loss = tf.reduce_mean(loss * weights)
            #loss = tf.reduce_sum(loss * weights)

            #loss = tfe.safe_divide(tf.reduce_sum(loss * weights), n_cls_positives, name='localization_loss')
            tf.losses.add_loss(loss)


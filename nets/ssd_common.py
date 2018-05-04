# Copyright 2015 Paul Balanca. All Rights Reserved.
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
"""Shared function between different SSD implementations.
"""
import numpy as np
import tensorflow as tf
import tf_extended as tfe


# =========================================================================== #
# TensorFlow implementation of boxes SSD encoding / decoding.
# =========================================================================== #
from tensorflow.contrib.image.python.ops import image_ops

def areas(bboxes):
    ymin, xmin, ymax, xmax = tf.split(bboxes, 4, axis=1)
    return (xmax - xmin) * (ymax - ymin)
def intersection(bboxes, gt_bboxes):
    ymin, xmin, ymax, xmax = tf.split(bboxes, 4, axis=1)
    gt_ymin, gt_xmin, gt_ymax, gt_xmax = [tf.transpose(b, perm=[1, 0]) for b in tf.split(gt_bboxes, 4, axis=1)]

    int_ymin = tf.maximum(ymin, gt_ymin)
    int_xmin = tf.maximum(xmin, gt_xmin)
    int_ymax = tf.minimum(ymax, gt_ymax)
    int_xmax = tf.minimum(xmax, gt_xmax)
    h = tf.maximum(int_ymax - int_ymin, 0.)
    w = tf.maximum(int_xmax - int_xmin, 0.)

    return h * w
def iou_matrix(bboxes, gt_bboxes):
    inter_vol = intersection(bboxes, gt_bboxes)
    union_vol = areas(bboxes) + tf.transpose(areas(gt_bboxes), perm=[1, 0]) - inter_vol

    return tf.where(tf.equal(union_vol, 0.0),
                    tf.zeros_like(inter_vol), tf.truediv(inter_vol, union_vol))

def do_dual_max_match(overlap_matrix, high_thres, low_thres, ignore_between = True, gt_max_first=True):
    '''
    overlap_matrix: num_gt * num_anchors
    '''
    anchors_to_gt = tf.argmax(overlap_matrix, axis=0)
    match_values = tf.reduce_max(overlap_matrix, axis=0)

    positive_mask = tf.greater(match_values, high_thres)
    less_mask = tf.less(match_values, low_thres)
    between_mask = tf.logical_and(tf.less(match_values, high_thres), tf.greater_equal(match_values, low_thres))
    negative_mask = less_mask if ignore_between else between_mask
    ignore_mask = between_mask if ignore_between else less_mask

    match_indices = tf.where(negative_mask, -1 * tf.ones_like(anchors_to_gt), anchors_to_gt)
    match_indices = tf.where(ignore_mask, -2 * tf.ones_like(match_indices), match_indices)

    anchors_to_gt_mask = tf.one_hot(tf.clip_by_value(match_indices, -1, tf.cast(tf.shape(overlap_matrix)[0], tf.int64)), tf.shape(overlap_matrix)[0], on_value=1, off_value=0, axis=0, dtype=tf.int32)

    gt_to_anchors = tf.argmax(overlap_matrix, axis=1)

    if gt_max_first:
        left_gt_to_anchors_mask = tf.one_hot(gt_to_anchors, tf.shape(overlap_matrix)[1], on_value=1, off_value=0, axis=1, dtype=tf.int32)
    else:
        left_gt_to_anchors_mask = tf.cast(tf.logical_and(tf.reduce_max(anchors_to_gt_mask, axis=1, keep_dims=True) < 1, tf.one_hot(gt_to_anchors, tf.shape(overlap_matrix)[1], on_value=True, off_value=False, axis=1, dtype=tf.bool)), tf.int64)

    selected_scores = tf.gather_nd(overlap_matrix, tf.stack([tf.where(tf.reduce_max(left_gt_to_anchors_mask, axis=0) > 0, tf.argmax(left_gt_to_anchors_mask, axis=0), anchors_to_gt), tf.range(tf.cast(tf.shape(overlap_matrix)[1], tf.int64))], axis=1))
    return tf.where(tf.reduce_max(left_gt_to_anchors_mask, axis=0) > 0, tf.argmax(left_gt_to_anchors_mask, axis=0), match_indices), selected_scores

def tf_ssd_bboxes_encode_layer(labels,
                               bboxes,
                               anchors_layer,
                               num_classes,
                               img_shape,
                               allowed_border,
                               no_annotation_label,
                               positive_threshold = 0.5,
                               ignore_threshold = 0.3,
                               prior_scaling = [0.1, 0.1, 0.2, 0.2],
                               dtype = tf.float32):
    """Encode groundtruth labels and bounding boxes using SSD anchors from
    one layer.

    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors_layer: Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores): Target Tensors.
    """
    # Anchors coordinates and volume.
    #print(positive_threshold, ignore_threshold)
    yref, xref, href, wref = anchors_layer

    ymin_ = yref - href / 2.
    xmin_ = xref - wref / 2.
    ymax_ = yref + href / 2.
    xmax_ = xref + wref / 2.
    ymin, xmin, ymax, xmax = tf.reshape(ymin_, [-1]), tf.reshape(xmin_, [-1]), tf.reshape(ymax_, [-1]), tf.reshape(xmax_, [-1])
    anchors_point = tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    inside_mask = tf.logical_and(tf.logical_and(ymin >= -allowed_border*1./img_shape[0],
                                                          xmin >= -allowed_border*1./img_shape[1]),
                                                          tf.logical_and(ymax < (img_shape[0] + allowed_border)*1./img_shape[0],
                                                          xmax < (img_shape[1] + allowed_border)*1./img_shape[1]))


    overlap_matrix = iou_matrix(bboxes, anchors_point) * tf.cast(tf.expand_dims(inside_mask, 0), tf.float32)


    matched_gt, feat_scores = do_dual_max_match(overlap_matrix, positive_threshold, ignore_threshold)

    matched_gt_mask = matched_gt > -1
    matched_indices = tf.clip_by_value(matched_gt, 0, tf.int64.max)
    feat_labels = tf.gather(labels, matched_indices)

    feat_ymin, feat_xmin, feat_ymax, feat_xmax = [tf.reshape(b, tf.shape(ymin_)) for b in tf.split(tf.gather(bboxes, matched_indices), 4, axis=1)]

    # Transform to center / size.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin

    bboxes = tf.stack([ymin_, xmin_, ymax_, xmax_], axis=-1)
    # Encode features.
    # the prior_scaling (in fact is 5 and 10) is use for balance the regression loss of center and with(or height)
    # (x-x_ref)/x_ref * 10 + log(w/w_ref) * 5
    feat_cy = (feat_cy - yref) / href / prior_scaling[0]
    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    feat_h = tf.log(feat_h / href) / prior_scaling[2]
    feat_w = tf.log(feat_w / wref) / prior_scaling[3]
    # Use SSD ordering: x / y / w / h instead of ours.
    feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    # now feat_localizations is our regression object

    return feat_labels * tf.cast(matched_gt_mask, tf.int64) + (-1 * tf.cast(matched_gt < -1, tf.int64)), tf.expand_dims(tf.reshape(tf.cast(matched_gt_mask, tf.float32), tf.shape(ymin_)), -1) * feat_localizations, feat_scores, bboxes

# def tf_ssd_bboxes_encode_layer(labels,
#                                bboxes,
#                                anchors_layer,
#                                num_classes,
#                                img_shape,
#                                allowed_border,
#                                no_annotation_label,
#                                positive_threshold = 0.5,
#                                ignore_threshold = 0.3,
#                                prior_scaling = [0.1, 0.1, 0.2, 0.2],
#                                dtype = tf.float32):
#     """Encode groundtruth labels and bounding boxes using SSD anchors from
#     one layer.

#     Arguments:
#       labels: 1D Tensor(int64) containing groundtruth labels;
#       bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
#       anchors_layer: Numpy array with layer anchors;
#       matching_threshold: Threshold for positive match with groundtruth bboxes;
#       prior_scaling: Scaling of encoded coordinates.

#     Return:
#       (target_labels, target_localizations, target_scores): Target Tensors.
#     """
#     # Anchors coordinates and volume.
#     #print(positive_threshold, ignore_threshold)
#     yref, xref, href, wref = anchors_layer
#     ymin = yref - href / 2.
#     xmin = xref - wref / 2.
#     ymax = yref + href / 2.
#     xmax = xref + wref / 2.
#     vol_anchors = (xmax - xmin) * (ymax - ymin)

#     inside_mask = tf.logical_and(tf.logical_and(tf.constant(ymin) >= -allowed_border*1./img_shape[0],
#                                                           tf.constant(xmin) >= -allowed_border*1./img_shape[1]),
#                                                           tf.logical_and(tf.constant(ymax) < (img_shape[0] + allowed_border)*1./img_shape[0],
#                                                           tf.constant(xmax) < (img_shape[1] + allowed_border)*1./img_shape[1]))

#     # Initialize tensors...
#     shape = (yref.shape[0], yref.shape[1], href.size)

#     feat_labels = tf.zeros(shape, dtype=tf.int64)
#     # store every jaccard score when loop, will update depends the score of anchor and current ground_truth
#     feat_scores = tf.zeros(shape, dtype=dtype)

#     feat_ymin = tf.zeros(shape, dtype=dtype)
#     feat_xmin = tf.zeros(shape, dtype=dtype)
#     feat_ymax = tf.ones(shape, dtype=dtype)
#     feat_xmax = tf.ones(shape, dtype=dtype)

#     max_mask = tf.cast(tf.zeros(shape, dtype=tf.int32), tf.bool)

#     def jaccard_with_anchors(bbox):
#         """Compute jaccard score between a box and the anchors.
#         """
#         # the inner square
#         int_ymin = tf.maximum(ymin, bbox[0])
#         int_xmin = tf.maximum(xmin, bbox[1])
#         int_ymax = tf.minimum(ymax, bbox[2])
#         int_xmax = tf.minimum(xmax, bbox[3])
#         h = tf.maximum(int_ymax - int_ymin, 0.)
#         w = tf.maximum(int_xmax - int_xmin, 0.)
#         # Volumes.
#         inter_vol = h * w
#         # vol_anchors - inter_vol -> each anchor's total size - inner_size
#         union_vol = vol_anchors - inter_vol \
#             + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
#         jaccard = tf.div(inter_vol, union_vol)
#         return jaccard

#     def intersection_with_anchors(bbox):
#         """Compute intersection between score a box and the anchors.
#         """
#         int_ymin = tf.maximum(ymin, bbox[0])
#         int_xmin = tf.maximum(xmin, bbox[1])
#         int_ymax = tf.minimum(ymax, bbox[2])
#         int_xmax = tf.minimum(xmax, bbox[3])
#         h = tf.maximum(int_ymax - int_ymin, 0.)
#         w = tf.maximum(int_xmax - int_xmin, 0.)
#         inter_vol = h * w
#         # scores measures how much of itself's size overlap with ground_truth
#         scores = tf.div(inter_vol, vol_anchors)
#         return scores

#     def condition(i, feat_labels, feat_scores,
#                   feat_ymin, feat_xmin, feat_ymax, feat_xmax, max_mask):
#         """Condition: check label index.
#         """
#         r = tf.less(i, tf.shape(labels))
#         return r[0]

#     def body(i, feat_labels, feat_scores,
#              feat_ymin, feat_xmin, feat_ymax, feat_xmax, max_mask):
#         """Body: update feature labels, scores and bboxes.
#         Follow the original SSD paper for that purpose:
#           - assign values when jaccard > 0.5;
#           - only update if beat the score of other bboxes.
#         """
#         # get i_th groud_truth(label && bbox)
#         label = labels[i]
#         bbox = bboxes[i]
#         # current ground_truth's overlap with all others' anchors
#         jaccard = tf.cast(inside_mask, dtype) * jaccard_with_anchors(bbox)
#         # the index of the max overlap for current ground_truth
#         max_jaccard = tf.maximum(tf.reduce_max(jaccard), ignore_threshold)
#         #max_jaccard = tf.Print(max_jaccard, [max_jaccard], message='max_jaccard: ', summarize=500)
#         all_cur_max_indice_mask = tf.equal(jaccard, max_jaccard)

#         choice_jaccard = tf.cast(all_cur_max_indice_mask, tf.float32) * jaccard * tf.random_uniform(tf.shape(all_cur_max_indice_mask), minval=1., maxval=10.)

#         max_choice_jaccard = tf.maximum(tf.reduce_max(choice_jaccard), ignore_threshold)
#         cur_max_indice_mask = tf.equal(choice_jaccard, max_choice_jaccard)
#         #new_max = tf.one_hot(tf.where(cur_max_indice_mask)[0, :-1], href.size, on_value=1, off_value=0, axis=-1)
#         #cur_max_indice_mask = tf.scatter_nd(tf.cast(tf.where(cur_max_indice_mask)[0], tf.int32), [1], tf.shape(jaccard))
#         #cur_max_indice_mask =  tf.cast(tf.zeros_like(tf.equal(jaccard, max_jaccard)), tf.bool)
#         #jaccard = tf.Print(jaccard, [tf.where(cur_max_indice_mask), href.size,tf.random_uniform(tf.shape(cur_max_indice_mask))], message='jaccard: ', summarize=100)

#         #cur_max_indice_mask = tf.Print(cur_max_indice_mask, [ tf.reduce_sum(tf.cast(cur_max_indice_mask, tf.float32)), cur_max_indice_mask ], message='cur_max_indice_mask: ', summarize=100)
#         # the locations where current overlap is higher than before
#         greater_than_current_mask = tf.greater(jaccard, feat_scores)
#         # we will update these locations as well as the current max_overlap location for this ground_truth
#         locations_to_update = tf.logical_or(greater_than_current_mask, cur_max_indice_mask)
#         # but we will ignore those locations where is the max_overlap for any ground_truth before
#         locations_to_update_with_mask = tf.logical_and(locations_to_update, tf.logical_not(max_mask))
#         # for current max_overlap
#         # for those current overlap is higher than before
#         # for those locations where is not the max_overlap for any before ground_truth
#         # update scores, so the terminal scores are either those max_overlap along the way or the max_overlap for any ground_truth
#         feat_scores = tf.where(locations_to_update_with_mask, jaccard, feat_scores)

#         # !!! because the difference of rules for score and label update !!!
#         # !!! so before we get the negtive examples we must use labels as positive mask first !!!
#         # for current max_overlap
#         # for current jaccard higher than before and higher than threshold
#         # for those locations where is not the max_overlap for any before ground_truth
#         # update labels, so the terminal labels are either those with max_overlap and higher than threshold along the way or the max_overlap for any ground_truth
#         # locations_to_update_labels = tf.logical_or(tf.greater(tf.cast(greater_than_current_mask, dtype) * jaccard, positive_threshold), cur_max_indice_mask)
#         locations_to_update_labels = tf.logical_and(tf.logical_or(tf.greater(tf.cast(greater_than_current_mask, dtype) * jaccard, positive_threshold), cur_max_indice_mask), tf.logical_not(max_mask))

#         #cur_max_indice_mask = tf.Print(cur_max_indice_mask, [tf.reduce_sum(tf.cast(cur_max_indice_mask, tf.float32))], message='cur_max_indice_mask: ', summarize=500)

#         locations_to_update_labels_mask = tf.cast(tf.logical_and(locations_to_update_labels, label < num_classes), dtype)

#         feat_labels = tf.cast(locations_to_update_labels_mask, tf.int64) * label + (1 - tf.cast(locations_to_update_labels_mask, tf.int64)) * feat_labels
#         #feat_scores = tf.where(mask, jaccard, feat_scores)
#         # update ground truth for each anchors depends on the mask
#         feat_ymin = locations_to_update_labels_mask * bbox[0] + (1 - locations_to_update_labels_mask) * feat_ymin
#         feat_xmin = locations_to_update_labels_mask * bbox[1] + (1 - locations_to_update_labels_mask) * feat_xmin
#         feat_ymax = locations_to_update_labels_mask * bbox[2] + (1 - locations_to_update_labels_mask) * feat_ymax
#         feat_xmax = locations_to_update_labels_mask * bbox[3] + (1 - locations_to_update_labels_mask) * feat_xmax

#         # update max_mask along the way
#         max_mask = tf.logical_or(max_mask, cur_max_indice_mask)

#         return [i+1, feat_labels, feat_scores,
#                 feat_ymin, feat_xmin, feat_ymax, feat_xmax, max_mask]
#     # Main loop definition.
#     # iterate betwween all ground_truth to encode anchors
#     i = 0
#     [i, feat_labels, feat_scores,
#      feat_ymin, feat_xmin,
#      feat_ymax, feat_xmax, max_mask] = tf.while_loop(condition, body,
#                                            [i, feat_labels, feat_scores,
#                                             feat_ymin, feat_xmin,
#                                             feat_ymax, feat_xmax, max_mask], parallel_iterations=16,
#                                                                             back_prop=False,
#                                                                             swap_memory=True)

#     inside_int_mask = tf.cast(inside_mask, tf.int64)
#     feat_labels =  (1 - inside_int_mask) * -1 + inside_int_mask * feat_labels
#     # Transform to center / size.
#     feat_cy = (feat_ymax + feat_ymin) / 2.
#     feat_cx = (feat_xmax + feat_xmin) / 2.
#     feat_h = feat_ymax - feat_ymin
#     feat_w = feat_xmax - feat_xmin
#     # Encode features.
#     # the prior_scaling (in fact is 5 and 10) is use for balance the regression loss of center and with(or height)
#     # (x-x_ref)/x_ref * 10 + log(w/w_ref) * 5
#     feat_cy = (feat_cy - yref) / href / prior_scaling[0]
#     feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
#     feat_h = tf.log(feat_h / href) / prior_scaling[2]
#     feat_w = tf.log(feat_w / wref) / prior_scaling[3]
#     # Use SSD ordering: x / y / w / h instead of ours.
#     feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
#     # now feat_localizations is our regression object
#     return feat_labels, feat_localizations, feat_scores


def tf_ssd_bboxes_encode(labels,
                         bboxes,
                         anchors,
                         num_classes,
                         img_shape,
                         allowed_borders,
                         no_annotation_label,
                         positive_threshold = 0.5,
                         ignore_threshold=0.3,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         dtype=tf.float32,
                         scope='ssd_bboxes_encode'):
    """Encode groundtruth labels and bounding boxes using SSD net anchors.
    Encoding boxes for all feature layers.

    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors: List of Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores):
        Each element is a list of target Tensors.
    """
    with tf.name_scope(scope):
        target_labels = []
        target_localizations = []
        target_scores = []
        target_bboxes = []

        shape_recorder = []
        full_shape_anchors = {}
        with tf.name_scope('anchor_concat'):
            for i, anchors_layer in enumerate(anchors):
                yref, xref, href, wref = anchors_layer

                ymin_ = yref - href / 2.
                xmin_ = xref - wref / 2.
                ymax_ = yref + href / 2.
                xmax_ = xref + wref / 2.

                shape_recorder.append(ymin_.shape)
                full_shape_yxhw = [(ymin_ + ymax_)/2, (xmin_ + xmax_)/2, (ymax_ - ymin_), (xmax_ - xmin_)]

                full_shape_anchors[i] = [np.reshape(_, (-1)) for _ in full_shape_yxhw]
            #print(full_shape_anchors)
            remap_anchors = list(zip(*full_shape_anchors.values()))

            for i in range(len(full_shape_anchors)):
                full_shape_anchors[i] = np.concatenate(remap_anchors[i], axis=0)
                #print(full_shape_anchors[i].shape)
            # print([_.shape for _ in remap_anchors[0]])
            # print([_.shape for _ in remap_anchors[1]])
            # print([_.shape for _ in remap_anchors[2]])
            # print([_.shape for _ in remap_anchors[3]])
            #print(shape_recorder)
        len_recorder = [np.prod(_) for _ in shape_recorder]
        #print(len_recorder)
        #print(allowed_borders)
        flaten_allowed_borders = []
        for i, allowed_border in enumerate(allowed_borders):
            flaten_allowed_borders.append([allowed_border]*len_recorder[i])
        #print([len(_) for _ in flaten_allowed_borders])
        flaten_allowed_borders = np.concatenate(flaten_allowed_borders, axis=0)

        t_labels, t_loc, t_scores, t_bbox = tf_ssd_bboxes_encode_layer(labels, bboxes, list(full_shape_anchors.values()), num_classes, img_shape, flaten_allowed_borders, no_annotation_label, positive_threshold, ignore_threshold, prior_scaling, dtype)

        reshaped_loc = []
        for i, loc in enumerate(tf.split(t_loc, len_recorder)):
            reshaped_loc.append(tf.reshape(loc, list(shape_recorder[i])+[-1]))
        reshaped_bbox = []
        for i, bbox in enumerate(tf.split(t_bbox, len_recorder)):
            reshaped_bbox.append(tf.reshape(bbox, list(shape_recorder[i])+[-1]))
        #print(reshaped_loc)
        #print(reshaped_bbox)
        return tf.split(t_labels, len_recorder), reshaped_loc, tf.split(t_scores, len_recorder), reshaped_bbox
    # with tf.name_scope(scope):
    #     target_labels = []
    #     target_localizations = []
    #     target_scores = []
    #     target_bboxes = []
    #     for i, anchors_layer in enumerate(anchors):
    #         with tf.name_scope('bboxes_encode_block_%i' % i):

    #             yref, xref, href, wref = anchors_layer

    #             ymin_ = yref - href / 2.
    #             xmin_ = xref - wref / 2.
    #             ymax_ = yref + href / 2.
    #             xmax_ = xref + wref / 2.

    #             yref_, xref_, href_, wref_ = (ymin_ + ymax_)/2, (xmin_ + xmax_)/2, (ymax_ - ymin_), (xmax_ - xmin_)
    #             t_labels, t_loc, t_scores, t_bbox = \
    #                 tf_ssd_bboxes_encode_layer(labels, bboxes, anchors_layer,
    #                                            num_classes, img_shape, allowed_borders[i], no_annotation_label,
    #                                            positive_threshold, ignore_threshold,
    #                                            prior_scaling, dtype)
    #             print('anchors_layer:', [yref_.shape, xref_.shape, href_.shape, wref_.shape])
    #             print('t_labels:', t_labels)
    #             print('t_loc:', t_loc)
    #             print('t_scores:', t_scores)
    #             print('t_bbox:', t_bbox)
    #             target_labels.append(t_labels)
    #             target_localizations.append(t_loc)
    #             target_scores.append(t_scores)
    #             target_bboxes.append(t_bbox)
    #     return target_labels, target_localizations, target_scores, target_bboxes


def tf_ssd_bboxes_decode_layer(feat_localizations,
                               anchors_layer,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Arguments:
      feat_localizations: Tensor containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      Tensor Nx4: ymin, xmin, ymax, xmax
    """
    yref, xref, href, wref = anchors_layer

    # Compute center, height and width
    cx = feat_localizations[:, :, :, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, :, :, 1] * href * prior_scaling[1] + yref
    w = wref * tf.exp(feat_localizations[:, :, :, :, 2] * prior_scaling[2])
    h = href * tf.exp(feat_localizations[:, :, :, :, 3] * prior_scaling[3])
    # Boxes coordinates.
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return bboxes


def tf_ssd_bboxes_decode(feat_localizations,
                         anchors,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         scope='ssd_bboxes_decode'):
    """Compute the relative bounding boxes from the SSD net features and
    reference anchors bounding boxes.

    Arguments:
      feat_localizations: List of Tensors containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      List of Tensors Nx4: ymin, xmin, ymax, xmax
    """
    with tf.name_scope(scope):
        bboxes = []
        for i, anchors_layer in enumerate(anchors):
            bboxes.append(
                tf_ssd_bboxes_decode_layer(feat_localizations[i],
                                           anchors_layer,
                                           prior_scaling))
        return bboxes


# =========================================================================== #
# SSD boxes selection.
# =========================================================================== #
def tf_ssd_bboxes_select_layer(predictions_layer, localizations_layer,
                               select_threshold=None,
                               num_classes=21,
                               ignore_class=0,
                               scope=None):
    """Extract classes, scores and bounding boxes from features in one layer.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_layer: A SSD prediction layer;
      localizations_layer: A SSD localization layer;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    select_threshold = 0.0 if select_threshold is None else select_threshold
    with tf.name_scope(scope, 'ssd_bboxes_select_layer',
                       [predictions_layer, localizations_layer]):
        # Reshape features: Batches x N x N_labels | 4
        p_shape = tfe.get_shape(predictions_layer)
        predictions_layer = tf.reshape(predictions_layer,
                                       tf.stack([p_shape[0], -1, p_shape[-1]]))
        l_shape = tfe.get_shape(localizations_layer)
        localizations_layer = tf.reshape(localizations_layer,
                                         tf.stack([l_shape[0], -1, l_shape[-1]]))
        # just consider those legal bboxes
        # localizations_mask = (localizations_layer[:, :, 0] < localizations_layer[:, :, 2])
        # localizations_mask = tf.logical_and(localizations_mask, (localizations_layer[:, :, 1] < localizations_layer[:, :, 3]))
        # localizations_mask = tf.Print(localizations_mask,[localizations_mask], message='localizations_mask: ', summarize=30)
        d_scores = {}
        d_bboxes = {}
        for c in range(0, num_classes):
            if c != ignore_class:
                # Remove boxes under the threshold.
                scores = predictions_layer[:, :, c]
                fmask = tf.cast(tf.greater(scores, select_threshold), scores.dtype)
                # fmask = tf.cast(tf.logical_and(tf.greater(scores, select_threshold), localizations_mask), scores.dtype)
                scores = scores * fmask
                bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)
                # Append to dictionary.
                d_scores[c] = scores
                d_bboxes[c] = bboxes

        return d_scores, d_bboxes


def tf_ssd_bboxes_select(predictions_net, localizations_net,
                         select_threshold=None,
                         num_classes=21,
                         ignore_class=0,
                         scope=None):
    """Extract classes, scores and bounding boxes from network output layers.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_net: List of SSD prediction layers;
      localizations_net: List of localization layers;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    with tf.name_scope(scope, 'ssd_bboxes_select',
                       [predictions_net, localizations_net]):
        l_scores = []
        l_bboxes = []
        for i in range(len(predictions_net)):
            scores, bboxes = tf_ssd_bboxes_select_layer(predictions_net[i],
                                                        localizations_net[i],
                                                        select_threshold,
                                                        num_classes,
                                                        ignore_class)
            l_scores.append(scores)
            l_bboxes.append(bboxes)
        # Concat results.
        d_scores = {}
        d_bboxes = {}
        for c in l_scores[0].keys():
            ls = [s[c] for s in l_scores]
            lb = [b[c] for b in l_bboxes]
            d_scores[c] = tf.concat(ls, axis=1)
            d_bboxes[c] = tf.concat(lb, axis=1)
        return d_scores, d_bboxes


def tf_ssd_bboxes_select_layer_all_classes(predictions_layer, localizations_layer,
                                           select_threshold=None):
    """Extract classes, scores and bounding boxes from features in one layer.
     Batch-compatible: inputs are supposed to have batch-type shapes.

     Args:
       predictions_layer: A SSD prediction layer;
       localizations_layer: A SSD localization layer;
      select_threshold: Classification threshold for selecting a box. If None,
        select boxes whose classification score is higher than 'no class'.
     Return:
      classes, scores, bboxes: Input Tensors.
     """
    # Reshape features: Batches x N x N_labels | 4
    p_shape = tfe.get_shape(predictions_layer)
    predictions_layer = tf.reshape(predictions_layer,
                                   tf.stack([p_shape[0], -1, p_shape[-1]]))
    l_shape = tfe.get_shape(localizations_layer)
    localizations_layer = tf.reshape(localizations_layer,
                                     tf.stack([l_shape[0], -1, l_shape[-1]]))
    # Boxes selection: use threshold or score > no-label criteria.
    if select_threshold is None or select_threshold == 0:
        # Class prediction and scores: assign 0. to 0-class
        classes = tf.argmax(predictions_layer, axis=2)
        scores = tf.reduce_max(predictions_layer, axis=2)
        scores = scores * tf.cast(classes > 0, scores.dtype)
    else:
        sub_predictions = predictions_layer[:, :, 1:]
        classes = tf.argmax(sub_predictions, axis=2) + 1
        scores = tf.reduce_max(sub_predictions, axis=2)
        # Only keep predictions higher than threshold.
        mask = tf.greater(scores, select_threshold)
        classes = classes * tf.cast(mask, classes.dtype)
        scores = scores * tf.cast(mask, scores.dtype)
    # Assume localization layer already decoded.
    bboxes = localizations_layer
    return classes, scores, bboxes


def tf_ssd_bboxes_select_all_classes(predictions_net, localizations_net,
                                     select_threshold=None,
                                     scope=None):
    """Extract classes, scores and bounding boxes from network output layers.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_net: List of SSD prediction layers;
      localizations_net: List of localization layers;
      select_threshold: Classification threshold for selecting a box. If None,
        select boxes whose classification score is higher than 'no class'.
    Return:
      classes, scores, bboxes: Tensors.
    """
    with tf.name_scope(scope, 'ssd_bboxes_select',
                       [predictions_net, localizations_net]):
        l_classes = []
        l_scores = []
        l_bboxes = []
        for i in range(len(predictions_net)):
            classes, scores, bboxes = \
                tf_ssd_bboxes_select_layer_all_classes(predictions_net[i],
                                                       localizations_net[i],
                                                       select_threshold)
            l_classes.append(classes)
            l_scores.append(scores)
            l_bboxes.append(bboxes)

        classes = tf.concat(l_classes, axis=1)
        scores = tf.concat(l_scores, axis=1)
        bboxes = tf.concat(l_bboxes, axis=1)
        return classes, scores, bboxes


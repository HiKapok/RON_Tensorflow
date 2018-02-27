import numpy as np
import sys
import os
import tensorflow as tf

slim = tf.contrib.slim


def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def vgg_16(inputs,
           is_training=True,
           scope='vgg_16'):
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    net = slim.max_pool2d(net, [3, 3], stride=1, padding='SAME', scope='pool5')
    # Use conv2d instead of fully_connected layers.
    net = slim.conv2d(net, 1024, [3, 3], stride=1, rate=6, padding='SAME', scope='fc6')
    net = slim.conv2d(net, 1024, [1, 1], stride=1, rate=1, padding='SAME', scope='fc7')

    return net

# reader = tf.train.NewCheckpointReader('./vgg16_reducedfc/tf_model/imagenet_vgg16_reducedfc.ckpt')
# print(reader.debug_string().decode("utf-8"))
'''load the IR model with renamed saver, then test the outputs and save with new variable names
'''
name_map = {
    'vgg_16/conv1/conv1_1/weights': 'ConvNd_0_weight',
    'vgg_16/conv1/conv1_1/biases': 'ConvNd_0_bias',
    'vgg_16/conv1/conv1_2/weights': 'ConvNd_2_weight',
    'vgg_16/conv1/conv1_2/biases': 'ConvNd_2_bias',
    'vgg_16/conv2/conv2_1/weights': 'ConvNd_5_weight',
    'vgg_16/conv2/conv2_1/biases': 'ConvNd_5_bias',
    'vgg_16/conv2/conv2_2/weights': 'ConvNd_7_weight',
    'vgg_16/conv2/conv2_2/biases': 'ConvNd_7_bias',
    'vgg_16/conv3/conv3_1/weights': 'ConvNd_10_weight',
    'vgg_16/conv3/conv3_1/biases': 'ConvNd_10_bias',
    'vgg_16/conv3/conv3_2/weights': 'ConvNd_12_weight',
    'vgg_16/conv3/conv3_2/biases': 'ConvNd_12_bias',
    'vgg_16/conv3/conv3_3/weights': 'ConvNd_14_weight',
    'vgg_16/conv3/conv3_3/biases': 'ConvNd_14_bias',
    'vgg_16/conv4/conv4_1/weights': 'ConvNd_17_weight',
    'vgg_16/conv4/conv4_1/biases': 'ConvNd_17_bias',
    'vgg_16/conv4/conv4_2/weights': 'ConvNd_19_weight',
    'vgg_16/conv4/conv4_2/biases': 'ConvNd_19_bias',
    'vgg_16/conv4/conv4_3/weights': 'ConvNd_21_weight',
    'vgg_16/conv4/conv4_3/biases': 'ConvNd_21_bias',
    'vgg_16/conv5/conv5_1/weights': 'ConvNd_24_weight',
    'vgg_16/conv5/conv5_1/biases': 'ConvNd_24_bias',
    'vgg_16/conv5/conv5_2/weights': 'ConvNd_26_weight',
    'vgg_16/conv5/conv5_2/biases': 'ConvNd_26_bias',
    'vgg_16/conv5/conv5_3/weights': 'ConvNd_28_weight',
    'vgg_16/conv5/conv5_3/biases': 'ConvNd_28_bias',
    'vgg_16/fc6/weights': 'ConvNd_31_weight',
    'vgg_16/fc6/biases': 'ConvNd_31_bias',
    'vgg_16/fc7/weights': 'ConvNd_33_weight',
    'vgg_16/fc7/biases': 'ConvNd_33_bias'}

# tf.reset_default_graph()

# input_image = tf.placeholder(tf.float32,  shape = (None, 224, 224, 3), name = 'input_placeholder')
# with slim.arg_scope(vgg_arg_scope()):
#     outputs = vgg_16(input_image, is_training=True)

# var_to_restore = {}
# for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):#TRAINABLE_VARIABLES):
#     var_name = var.op.name
#     #print(var_name)
#     var_to_restore[name_map[var_name]] = var

# saver_restore = tf.train.Saver(var_to_restore)
# saver = tf.train.Saver()

# with tf.Session() as sess:

#     init = tf.global_variables_initializer()
#     sess.run(init)

#     saver_restore.restore(sess, "./vgg16_reducedfc/tf_model/imagenet_vgg16_reducedfc.ckpt")

#     predict = np.transpose(sess.run(outputs, feed_dict = {input_image : np.ones((1,224,224,3)) * 0.5}), (0,3,1,2))
#     print(predict)
#     print(np.argmax(predict))

#     save_path = saver.save(sess, "./vgg16_reducedfc/tf_renamed_model/vgg16_reducedfc.ckpt")
#     print("Model saved in path: %s" % save_path)


'''run test for the renamed chcekpoint again
'''

# tf.reset_default_graph()

input_image = tf.placeholder(tf.float32,  shape = (None, 224, 224, 3), name = 'input_placeholder')
with slim.arg_scope(vgg_arg_scope()):
    outputs = vgg_16(input_image, is_training=True)

saver = tf.train.Saver()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    saver.restore(sess, "./vgg16_reducedfc/tf_renamed_model/vgg16_reducedfc.ckpt")

    predict = np.transpose(sess.run(outputs, feed_dict = {input_image : np.ones((1,224,224,3)) * 0.5}), (0,3,1,2))
    print(predict)
    print(np.argmax(predict))



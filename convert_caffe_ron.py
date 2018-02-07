#convert from caffe
# https://github.com/hujie-frank/SENet
# https://github.com/ruotianluo/pytorch-resnet
# /ruotianluo/pytorch-resnet/master/convert.py
#
# install caffe python 3.6
#   https://yangcha.github.io/Caffe-Conda3/

# import caffe
import sys
import os
sys.path.insert(0,'/media/rs/7A0EE8880EE83EAF1/Kapok/caffe-master/python')
os.environ["GLOG_minloglevel"] = "2"
import caffe
from caffe.proto import caffe_pb2


# others
import re
import numpy as np
from collections import OrderedDict
import cv2




##--------------------------------------------------------

# how to get caffe keys:
#   caffe_model.params.keys()
#   odict_keys(['conv1/7x7_s2', 'conv1/7x7_s2/bn', 'conv1/7x7_s2/bn/scale', 'conv2_1_1x1_reduce', 'conv2_1_1x1_reduce/bn', 'conv2_1_1x1_reduce/bn/scale', 'conv2_1_3x3', 'conv2_1_3x3/bn', 'conv2_1_3x3/bn/scale', 'conv2_1_1x1_increase', 'conv2_1_1x1_increase/bn', 'conv2_1_1x1_increase/bn/scale', 'conv2_1_1x1_down', 'conv2_1_1x1_up', 'conv2_1_1x1_proj', 'conv2_1_1x1_proj/bn', 'conv2_1_1x1_proj/bn/scale', 'conv2_2_1x1_reduce', 'conv2_2_1x1_reduce/bn', 'conv2_2_1x1_reduce/bn/scale', 'conv2_2_3x3', 'conv2_2_3x3/bn', 'conv2_2_3x3/bn/scale', 'conv2_2_1x1_increase', 'conv2_2_1x1_increase/bn', 'conv2_2_1x1_increase/bn/scale', 'conv2_2_1x1_down', 'conv2_2_1x1_up', 'conv2_3_1x1_reduce', 'conv2_3_1x1_reduce/bn', 'conv2_3_1x1_reduce/bn/scale', 'conv2_3_3x3', 'conv2_3_3x3/bn', 'conv2_3_3x3/bn/scale', 'conv2_3_1x1_increase', 'conv2_3_1x1_increase/bn', 'conv2_3_1x1_increase/bn/scale', 'conv2_3_1x1_down', 'conv2_3_1x1_up', 'conv3_1_1x1_reduce', 'conv3_1_1x1_reduce/bn', 'conv3_1_1x1_reduce/bn/scale', 'conv3_1_3x3', 'conv3_1_3x3/bn', 'conv3_1_3x3/bn/scale', 'conv3_1_1x1_increase', 'conv3_1_1x1_increase/bn', 'conv3_1_1x1_increase/bn/scale', 'conv3_1_1x1_down', 'conv3_1_1x1_up', 'conv3_1_1x1_proj', 'conv3_1_1x1_proj/bn', 'conv3_1_1x1_proj/bn/scale', 'conv3_2_1x1_reduce', 'conv3_2_1x1_reduce/bn', 'conv3_2_1x1_reduce/bn/scale', 'conv3_2_3x3', 'conv3_2_3x3/bn', 'conv3_2_3x3/bn/scale', 'conv3_2_1x1_increase', 'conv3_2_1x1_increase/bn', 'conv3_2_1x1_increase/bn/scale', 'conv3_2_1x1_down', 'conv3_2_1x1_up', 'conv3_3_1x1_reduce', 'conv3_3_1x1_reduce/bn', 'conv3_3_1x1_reduce/bn/scale', 'conv3_3_3x3', 'conv3_3_3x3/bn', 'conv3_3_3x3/bn/scale', 'conv3_3_1x1_increase', 'conv3_3_1x1_increase/bn', 'conv3_3_1x1_increase/bn/scale', 'conv3_3_1x1_down', 'conv3_3_1x1_up', 'conv3_4_1x1_reduce', 'conv3_4_1x1_reduce/bn', 'conv3_4_1x1_reduce/bn/scale', 'conv3_4_3x3', 'conv3_4_3x3/bn', 'conv3_4_3x3/bn/scale', 'conv3_4_1x1_increase', 'conv3_4_1x1_increase/bn', 'conv3_4_1x1_increase/bn/scale', 'conv3_4_1x1_down', 'conv3_4_1x1_up', 'conv4_1_1x1_reduce', 'conv4_1_1x1_reduce/bn', 'conv4_1_1x1_reduce/bn/scale', 'conv4_1_3x3', 'conv4_1_3x3/bn', 'conv4_1_3x3/bn/scale', 'conv4_1_1x1_increase', 'conv4_1_1x1_increase/bn', 'conv4_1_1x1_increase/bn/scale', 'conv4_1_1x1_down', 'conv4_1_1x1_up', 'conv4_1_1x1_proj', 'conv4_1_1x1_proj/bn', 'conv4_1_1x1_proj/bn/scale', 'conv4_2_1x1_reduce', 'conv4_2_1x1_reduce/bn', 'conv4_2_1x1_reduce/bn/scale', 'conv4_2_3x3', 'conv4_2_3x3/bn', 'conv4_2_3x3/bn/scale', 'conv4_2_1x1_increase', 'conv4_2_1x1_increase/bn', 'conv4_2_1x1_increase/bn/scale', 'conv4_2_1x1_down', 'conv4_2_1x1_up', 'conv4_3_1x1_reduce', 'conv4_3_1x1_reduce/bn', 'conv4_3_1x1_reduce/bn/scale', 'conv4_3_3x3', 'conv4_3_3x3/bn', 'conv4_3_3x3/bn/scale', 'conv4_3_1x1_increase', 'conv4_3_1x1_increase/bn', 'conv4_3_1x1_increase/bn/scale', 'conv4_3_1x1_down', 'conv4_3_1x1_up', 'conv4_4_1x1_reduce', 'conv4_4_1x1_reduce/bn', 'conv4_4_1x1_reduce/bn/scale', 'conv4_4_3x3', 'conv4_4_3x3/bn', 'conv4_4_3x3/bn/scale', 'conv4_4_1x1_increase', 'conv4_4_1x1_increase/bn', 'conv4_4_1x1_increase/bn/scale', 'conv4_4_1x1_down', 'conv4_4_1x1_up', 'conv4_5_1x1_reduce', 'conv4_5_1x1_reduce/bn', 'conv4_5_1x1_reduce/bn/scale', 'conv4_5_3x3', 'conv4_5_3x3/bn', 'conv4_5_3x3/bn/scale', 'conv4_5_1x1_increase', 'conv4_5_1x1_increase/bn', 'conv4_5_1x1_increase/bn/scale', 'conv4_5_1x1_down', 'conv4_5_1x1_up', 'conv4_6_1x1_reduce', 'conv4_6_1x1_reduce/bn', 'conv4_6_1x1_reduce/bn/scale', 'conv4_6_3x3', 'conv4_6_3x3/bn', 'conv4_6_3x3/bn/scale', 'conv4_6_1x1_increase', 'conv4_6_1x1_increase/bn', 'conv4_6_1x1_increase/bn/scale', 'conv4_6_1x1_down', 'conv4_6_1x1_up', 'conv5_1_1x1_reduce', 'conv5_1_1x1_reduce/bn', 'conv5_1_1x1_reduce/bn/scale', 'conv5_1_3x3', 'conv5_1_3x3/bn', 'conv5_1_3x3/bn/scale', 'conv5_1_1x1_increase', 'conv5_1_1x1_increase/bn', 'conv5_1_1x1_increase/bn/scale', 'conv5_1_1x1_down', 'conv5_1_1x1_up', 'conv5_1_1x1_proj', 'conv5_1_1x1_proj/bn', 'conv5_1_1x1_proj/bn/scale', 'conv5_2_1x1_reduce', 'conv5_2_1x1_reduce/bn', 'conv5_2_1x1_reduce/bn/scale', 'conv5_2_3x3', 'conv5_2_3x3/bn', 'conv5_2_3x3/bn/scale', 'conv5_2_1x1_increase', 'conv5_2_1x1_increase/bn', 'conv5_2_1x1_increase/bn/scale', 'conv5_2_1x1_down', 'conv5_2_1x1_up', 'conv5_3_1x1_reduce', 'conv5_3_1x1_reduce/bn', 'conv5_3_1x1_reduce/bn/scale', 'conv5_3_3x3', 'conv5_3_3x3/bn', 'conv5_3_3x3/bn/scale', 'conv5_3_1x1_increase', 'conv5_3_1x1_increase/bn', 'conv5_3_1x1_increase/bn/scale', 'conv5_3_1x1_down', 'conv5_3_1x1_up', 'classifier'])
#

# how to copy weights:
# e.g.
#   pytorch_state_dict['conv.weight'] = caffe_net_params['conv'][0].data
#   pytorch_state_dict['conv.bias  '] = caffe_net_params['conv'][1].data



# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ...' % os.path.basename(__file__))
    prototxt_file = '/media/rs/3EBAC1C7BAC17BC1/Detections/RON_Tensorflow/traincudnn.prototxt'
    caffemodel_file = '/media/rs/3EBAC1C7BAC17BC1/Detections/RON_Tensorflow/model/RON320_VOC0712_VOC07.caffemodel'

    caffe.set_mode_cpu()
    caffe_net = caffe.Net(prototxt_file, caffemodel_file, caffe.TEST)

    caffe_net_params = caffe_net.params
    print(caffe_model.params.keys())

    exit(0)










# RON in TensorFlow: Reverse Connection with Objectness Prior Networks for Object Detection 

RON is an object detection system for efficient object detection framework as descibed in [This CVPR paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Kong_RON_Reverse_Connection_CVPR_2017_paper.pdf). 

This repository contains code of the re-implement of RON following the above paper.

The code is modified from [SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow). You can use the code to train/evaluate a network for object detection task. 
For more details, please refer to [README of SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow/blob/master/README.md).
##  ##
update:

- Add SSD preprocesing method using Tensorflow
- Modify the network to match the original Caffe code
- Add nms using Tensorflow ops to support two mode
- Replica GPU training support (If you are using Tensorflow 1.5.0+, then remove the replicate_model\_fn.py)
- Add voc eval
- Add realtime eval, using class-wise bboxes select and nms
- Add training use vgg16_reducedfc model converted from pytorch, you can get from [this](https://drive.google.com/open?id=184srhbt8_uvLKeWW_Yo8Mc5wTyc0lJT7)
- Other fixes

Note: Training is in process, and the model will be released later.

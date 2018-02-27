import keras  # work around segfault
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from mmdnn.conversion.keras.keras2_parser import Keras2Parser


class VGG16(nn.Module):
    def __init__(self, base):
        super(VGG16, self).__init__()
        self.vgg = nn.ModuleList(base)
        print('layers num in vgg16_reducedfc', len(self.vgg))

    def forward(self, x):
        # apply vgg up to conv4_3 relu
        for k in range(35):
            x = self.vgg[k](x)
        return x

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

from pytorch2keras.converter import pytorch_to_keras

def convert_pytorch2keras2ir():
    max_error = 0
    model = VGG16(vgg([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512], 3))
    # load weights here
    state_dict_to_load = torch.load('vgg16_reducedfc/vgg16_reducedfc.pth', map_location=lambda storage, loc: storage)
    renamed_tate_dict_to_load = {}
    for k, v in state_dict_to_load.items():
        renamed_tate_dict_to_load['vgg.' + k] = v
    print(model.state_dict().keys())
    print(state_dict_to_load.keys())

    model.load_state_dict(renamed_tate_dict_to_load)

    for m in model.modules():
        m.training = False

    input_np = np.ones((1, 3, 224, 224)) * 0.5
    # np.random.uniform(0, 1, (1, 3, 224, 224))
    input_var = Variable(torch.FloatTensor(input_np))
    output = model(input_var)

    k_model = pytorch_to_keras((3, 224, 224,), output)

    pytorch_output = output.data.numpy()

    print(pytorch_output)
    print(np.argmax(pytorch_output))
    return

    keras_output = k_model.predict(input_np)

    error = np.max(pytorch_output - keras_output)
    print(error)
    if max_error < error:
        max_error = error

    print('Max error: {0}'.format(max_error))

    # save network structure as JSON
    json_string = k_model.to_json()
    with open("vgg16_reducedfc/imagenet_vgg16_reducedfc.json", "w") as of:
        of.write(json_string)

    print("Network structure is saved as [vgg16_reducedfc/imagenet_vgg16_reducedfc.json].")

    k_model.save_weights('vgg16_reducedfc/imagenet_vgg16_reducedfc.h5')

    print("Network weights are saved as [vgg16_reducedfc/imagenet_vgg16_reducedfc.h5].")

    parser = Keras2Parser(('vgg16_reducedfc/imagenet_vgg16_reducedfc.json', 'vgg16_reducedfc/imagenet_vgg16_reducedfc.h5'))
    parser.run('vgg16_reducedfc/ir')

if __name__ == '__main__':
    pass
    #convert_pytorch2keras2ir()
#python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath vgg16_reducedfc/ir.pb --IRWeightPath vgg16_reducedfc/ir.npy --dstModelPath vgg16_reducedfc/tf_vgg16.py

import numpy as np
import sys
import os
import tensorflow as tf
import tf_vgg16


input_placeholder, output = tf_vgg16.KitModel('./vgg16_reducedfc/ir.npy')

for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print(var.op.name)
saver = tf.train.Saver()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    predict = np.transpose(sess.run(output, feed_dict = {input_placeholder : np.ones((1, 224, 224, 3)) * 0.5}), (0,3,1,2))
    print(predict)
    print(np.argmax(predict))

    save_path = saver.save(sess, "./vgg16_reducedfc/tf_model/imagenet_vgg16_reducedfc.ckpt")
    print("Model saved in path: %s" % save_path)

# layers num in vgg16_reducedfc 35
# odict_keys(['vgg.0.weight', 'vgg.0.bias', 'vgg.2.weight', 'vgg.2.bias', 'vgg.5.weight', 'vgg.5.bias', 'vgg.7.weight', 'vgg.7.bias', 'vgg.10.weight', 'vgg.10.bias', 'vgg.12.weight', 'vgg.12.bias', 'vgg.14.weight', 'vgg.14.bias', 'vgg.17.weight', 'vgg.17.bias', 'vgg.19.weight', 'vgg.19.bias', 'vgg.21.weight', 'vgg.21.bias', 'vgg.24.weight', 'vgg.24.bias', 'vgg.26.weight', 'vgg.26.bias', 'vgg.28.weight', 'vgg.28.bias', 'vgg.31.weight', 'vgg.31.bias', 'vgg.33.weight', 'vgg.33.bias'])
# odict_keys(['0.weight', '0.bias', '2.weight', '2.bias', '5.weight', '5.bias', '7.weight', '7.bias', '10.weight', '10.bias', '12.weight', '12.bias', '14.weight', '14.bias', '17.weight', '17.bias', '19.weight', '19.bias', '21.weight', '21.bias', '24.weight', '24.bias', '26.weight', '26.bias', '28.weight', '28.bias', '31.weight', '31.bias', '33.weight', '33.bias'])
# Queued 34, processing ConvNd_0
# Converting convolution ...
# Queued 33, processing Threshold_1
# Converting RELU ...
# Queued 32, processing ConvNd_2
# Converting convolution ...
# Queued 31, processing Threshold_3
# Converting RELU ...
# Queued 30, processing MaxPool2d_4
# Converting pooling ...
# Queued 29, processing ConvNd_5
# Converting convolution ...
# Queued 28, processing Threshold_6
# Converting RELU ...
# Queued 27, processing ConvNd_7
# Converting convolution ...
# Queued 26, processing Threshold_8
# Converting RELU ...
# Queued 25, processing MaxPool2d_9
# Converting pooling ...
# Queued 24, processing ConvNd_10
# Converting convolution ...
# Queued 23, processing Threshold_11
# Converting RELU ...
# Queued 22, processing ConvNd_12
# Converting convolution ...
# Queued 21, processing Threshold_13
# Converting RELU ...
# Queued 20, processing ConvNd_14
# Converting convolution ...
# Queued 19, processing Threshold_15
# Converting RELU ...
# Queued 18, processing MaxPool2d_16
# Converting pooling ...
# Queued 17, processing ConvNd_17
# Converting convolution ...
# Queued 16, processing Threshold_18
# Converting RELU ...
# Queued 15, processing ConvNd_19
# Converting convolution ...
# Queued 14, processing Threshold_20
# Converting RELU ...
# Queued 13, processing ConvNd_21
# Converting convolution ...
# Queued 12, processing Threshold_22
# Converting RELU ...
# Queued 11, processing MaxPool2d_23
# Converting pooling ...
# Queued 10, processing ConvNd_24
# Converting convolution ...
# Queued 9, processing Threshold_25
# Converting RELU ...
# Queued 8, processing ConvNd_26
# Converting convolution ...
# Queued 7, processing Threshold_27
# Converting RELU ...
# Queued 6, processing ConvNd_28
# Converting convolution ...
# Queued 5, processing Threshold_29
# Converting RELU ...
# Queued 4, processing MaxPool2d_30
# Converting pooling ...
# Queued 3, processing ConvNd_31
# Converting convolution ...
# Queued 2, processing Threshold_32
# Converting RELU ...
# Queued 1, processing ConvNd_33
# Converting convolution ...
# Queued 0, processing Threshold_34
# Converting RELU ...
# [[[[0.56518674 0.5648855  0.56539273 ... 0.55910426 0.55902666
#     0.55963665]
#    [0.5646307  0.56441396 0.56490076 ... 0.55833876 0.558248
#     0.5587633 ]
#    [0.56682736 0.56642884 0.56690675 ... 0.56059074 0.5606095
#     0.561705  ]
#    ...
#    [0.56652933 0.566047   0.5669157  ... 0.5684799  0.56791127
#     0.5686564 ]
#    [0.56783754 0.5674196  0.56859696 ... 0.56904054 0.5681525
#     0.56862277]
#    [0.56805956 0.56767094 0.5687252  ... 0.5693993  0.5685856
#     0.5689803 ]]

#   [[0.61160606 0.61197567 0.61136156 ... 0.6035788  0.60422885
#     0.60461605]
#    [0.6118964  0.61229753 0.6115499  ... 0.60425526 0.6047989
#     0.6048801 ]
#    [0.6115182  0.61181307 0.6112713  ... 0.60355365 0.6039314
#     0.60379446]
#    ...
#    [0.62035894 0.6207421  0.6202834  ... 0.6248009  0.6254995
#     0.6250031 ]
#    [0.62169003 0.621992   0.6213436  ... 0.6256129  0.62641656
#     0.6257957 ]
#    [0.6220664  0.6223428  0.62193173 ... 0.6261158  0.62668806
#     0.6259797 ]]

#   [[0.65719277 0.6575201  0.6574769  ... 0.65573096 0.6545163
#     0.65423983]
#    [0.6577686  0.65782446 0.65776503 ... 0.65566814 0.65450025
#     0.65431076]
#    [0.65860355 0.6586724  0.6584081  ... 0.6561669  0.6555219
#     0.655506  ]
#    ...
#    [0.644818   0.6450008  0.64469594 ... 0.6414154  0.63991183
#     0.6402248 ]
#    [0.6446375  0.6446437  0.6449769  ... 0.6413419  0.6401791
#     0.6400857 ]
#    [0.64427376 0.6443268  0.6448113  ... 0.64124846 0.63980645
#     0.63967323]]

#   ...

#   [[0.48124567 0.48095548 0.48078826 ... 0.47762334 0.478104
#     0.47843274]
#    [0.48163638 0.4815083  0.48161095 ... 0.47901338 0.47898895
#     0.47929147]
#    [0.48117283 0.4810738  0.48140502 ... 0.47886458 0.4787247
#     0.47904274]
#    ...
#    [0.49446964 0.4948662  0.49492195 ... 0.48682523 0.4872975
#     0.48713356]
#    [0.49738318 0.49736002 0.49746874 ... 0.48887727 0.48895437
#     0.4889008 ]
#    [0.49715272 0.4970457  0.49713236 ... 0.48862988 0.4888101
#     0.48880836]]

#   [[0.56949276 0.56935173 0.5684848  ... 0.569404   0.57013416
#     0.570288  ]
#    [0.57003045 0.56991214 0.5692532  ... 0.57016397 0.5707201
#     0.5708895 ]
#    [0.5702013  0.57008487 0.5697614  ... 0.5707301  0.57091165
#     0.5711077 ]
#    ...
#    [0.5794762  0.57988876 0.5787656  ... 0.5872504  0.587631
#     0.5872952 ]
#    [0.58056194 0.58079237 0.57949674 ... 0.5882771  0.58854294
#     0.5882966 ]
#    [0.5807397  0.5809685  0.57967806 ... 0.5882982  0.5886503
#     0.5883287 ]]

#   [[0.1272643  0.12772961 0.12823    ... 0.12941207 0.1295461
#     0.12953535]
#    [0.12778667 0.1282799  0.12920344 ... 0.1303485  0.13025537
#     0.1299596 ]
#    [0.1277322  0.12831642 0.12927507 ... 0.13005951 0.12988716
#     0.12942441]
#    ...
#    [0.12598167 0.1265063  0.12763491 ... 0.11985718 0.11855377
#     0.11838106]
#    [0.12606645 0.12659647 0.12803979 ... 0.12096258 0.11957263
#     0.11933898]
#    [0.12525684 0.12579326 0.12718993 ... 0.12005685 0.11878219
#     0.11868025]]]]
# 115951
# 3.5762787e-07
# Max error: 3.5762786865234375e-07
# Network structure is saved as [vgg16_reducedfc/imagenet_vgg16_reducedfc.json].
# Network weights are saved as [vgg16_reducedfc/imagenet_vgg16_reducedfc.h5].
# Network file [vgg16_reducedfc/imagenet_vgg16_reducedfc.json] and [vgg16_reducedfc/imagenet_vgg16_reducedfc.h5] is loaded successfully.
# IR network structure is saved as [vgg16_reducedfc/ir.json].
# IR network structure is saved as [vgg16_reducedfc/ir.pb].
# IR weights are saved as [vgg16_reducedfc/ir.npy].

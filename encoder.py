# Encoder is fixed to the first few layers
# of VGG-19 (pre-trained on ImageNet)

import numpy as np
import tensorflow as tf


ENCODER_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1',
)


class Encoder(object):

    def __init__(self, weights_path, block_id):
        # load weights (kernel and bias) from npz file
        weights = np.load(weights_path)

        idx = 0
        self.weight_vars = []
        self.output_layer = 'relu%d_1' % block_id

        # create the TensorFlow variables
        encoder_name = 'encoder%d' % block_id

        with tf.variable_scope(encoder_name):
            for layer in ENCODER_LAYERS:
                kind = layer[:4]

                if kind == 'conv':
                    kernel = weights['arr_%d' % idx].transpose([2, 3, 1, 0])
                    bias   = weights['arr_%d' % (idx + 1)]
                    kernel = kernel.astype(np.float32)
                    bias   = bias.astype(np.float32)
                    idx   += 2

                    with tf.variable_scope(layer):
                        W = tf.Variable(kernel, trainable=False, name='kernel')
                        b = tf.Variable(bias,   trainable=False, name='bias')

                    self.weight_vars.append((W, b))

    def encode(self, image):
        # create the computational graph
        idx = 0
        out = image

        # switch RGB to BGR
        out = tf.reverse(out, axis=[-1])

        # preprocess(centralize) image
        out = preprocess(out)

        for layer in ENCODER_LAYERS:
            kind = layer[:4]

            if kind == 'conv':
                kernel, bias = self.weight_vars[idx]
                idx += 1
                out = conv2d(out, kernel, bias)

            elif kind == 'relu':
                out = tf.nn.relu(out)

            elif kind == 'pool':
                out = pool2d(out)

            if layer == self.output_layer:
                break

        return out


def preprocess(image, mode='BGR'):
    if mode == 'BGR':
        return image - np.array([103.939, 116.779, 123.68])
    else:
        return image - np.array([123.68, 116.779, 103.939])


def conv2d(x, kernel, bias):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)

    return out


def pool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


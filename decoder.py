# Decoder mostly mirrors the encoder with all pooling layers replaced by nearest
# up-sampling to reduce checker-board effects.

import numpy as np
import tensorflow as tf


WEIGHT_INIT_STDDEV = 0.1

DECODER_LAYERS = {
    'conv5_1' : (512, 512, 3),

    'conv4_4' : (512, 512, 3),
    'conv4_3' : (512, 512, 3),
    'conv4_2' : (512, 512, 3),
    'conv4_1' : (512, 256, 3),

    'conv3_4' : (256, 256, 3),
    'conv3_3' : (256, 256, 3),
    'conv3_2' : (256, 256, 3),
    'conv3_1' : (256, 128, 3),

    'conv2_2' : (128, 128, 3),
    'conv2_1' : (128,  64, 3),

    'conv1_2' : (64, 64, 3),
    'conv1_1' : (64,  3, 3),
}

UPSAMPLING_LAYERS = (
    'conv5_1', 'conv4_1', 
    'conv3_1', 'conv2_1',
)


class Decoder(object):

    def __init__(self, layers):
        self.weight_vars = []
        self.upsample_indices = []

        decoder_name = 'decoder%s' % layers[0][4]

        with tf.variable_scope(decoder_name):
            for idx in range(len(layers)):
                input_filters, output_filters, kernel_size = DECODER_LAYERS[layers[idx]]

                self.weight_vars.append(
                        self._create_variables(input_filters, 
                                               output_filters, 
                                               kernel_size, 
                                               scope=layers[idx]))

                if layers[idx] in UPSAMPLING_LAYERS:
                    self.upsample_indices.append(idx)

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        with tf.variable_scope(scope):
            shape  = [kernel_size, kernel_size, input_filters, output_filters]
            # kernel = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='kernel')
            # bias   = tf.Variable(tf.zeros([output_filters]), name='bias')
            kernel = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), 
                shape=shape, name='kernel', trainable=True)
            bias = tf.get_variable(initializer=tf.zeros([output_filters]), 
                name='bias', trainable=True)
            return (kernel, bias)

    def decode(self, image):
        final_layer_idx  = len(self.weight_vars) - 1

        out = image
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]

            if i == final_layer_idx:
                out = conv2d(out, kernel, bias, use_relu=False)
            else:
                out = conv2d(out, kernel, bias)
            
            if i in self.upsample_indices:
                out = upsample(out)

        # deprocess image
        out = deprocess(out)

        # switch BGR back to RGB
        out = tf.reverse(out, axis=[-1])

        # clip to 0..255
        out = tf.clip_by_value(out, 0.0, 255.0)

        return out


def deprocess(image, mode='BGR'):
    if mode == 'BGR':
        return image + np.array([103.939, 116.779, 123.68])
    else:
        return image + np.array([123.68, 116.779, 103.939])


def conv2d(x, kernel, bias, use_relu=True):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)

    if use_relu:
        out = tf.nn.relu(out)

    return out


def upsample(x, scale=2):
    height = tf.shape(x)[1] * scale
    width  = tf.shape(x)[2] * scale
    output = tf.image.resize_images(x, [height, width], 
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return output


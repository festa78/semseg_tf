"""Common functionals for network definition.
"""

import numpy as np
import tensorflow as tf


def make_conv_weights(shape, std):
    init_op = tf.truncated_normal_initializer(stddev=std)
    return tf.get_variable('weights', shape=shape, initializer=init_op)


def make_conv_biases(out_channels, constant):
    init_op = tf.constant_initializer(constant)
    return tf.get_variable(
        name='biases', shape=[out_channels], initializer=init_op)


def make_conv2d(out_channels, kernel_size, stride=1, bias=True, name=""):
    strides = [1, stride, stride, 1]

    def conv2d(in_features):
        with tf.variable_scope(name):
            in_channels = in_features.get_shape()[3].value
            # He initialization.
            std = (2. / in_channels)**.5
            weights_shape = [
                kernel_size, kernel_size, in_channels, out_channels
            ]
            weights = make_conv_weights(weights_shape, std)
            conv = tf.nn.conv2d(in_features, weights, strides, padding='SAME')
            if bias is True:
                biases = make_conv_biases(out_channels, 0.)
                conv = tf.nn.bias_add(conv, biases)
        return conv

    return conv2d


def make_atrous_conv2d(out_channels, kernel_size, atrous_rate, bias, name):

    def atrous_conv2d(in_features):
        with tf.variable_scope(name):
            in_channels = in_features.get_shape()[3].value
            # He initialization.
            std = (2. / in_channels)**.5
            weights_shape = [
                kernel_size, kernel_size, in_channels, out_channels
            ]
            weights = make_conv_weights(weights_shape, std)

            conv = tf.nn.atrous_conv2d(
                in_features, weights, atrous_rate, padding='SAME')
            if bias is True:
                biases = make_conv_biases(out_channels, 0.)
                conv = tf.nn.bias_add(conv, biases)
        return conv

    return atrous_conv2d


def make_avg_pool2d(kernel_size, stride, name):
    kernel_sizes = [1, kernel_size, kernel_size, 1]
    strides = [1, stride, stride, 1]

    def avg_pool2d(in_features):
        with tf.variable_scope(name):
            pool = tf.nn.avg_pool(
                in_features, kernel_sizes, strides, padding='SAME')
        return pool

    return avg_pool2d


def make_max_pool2d(kernel_size, stride, name):
    kernel_sizes = [1, kernel_size, kernel_size, 1]
    strides = [1, stride, stride, 1]

    def max_pool2d(in_features):
        with tf.variable_scope(name):
            pool = tf.nn.max_pool(
                in_features, kernel_sizes, strides, padding='SAME')
        return pool

    return max_pool2d


def make_bn2d(name):

    def bn2d(in_features):
        with tf.variable_scope(name):
            bn = tf.layers.batch_normalization(
                in_features,
                momentum=0.95,
                epsilon=1.e-5,
                training=self.training)
        return bn

    return bn2d


def make_concat(name):

    def concat(in_features_list):
        with tf.variable_scope(name):
            cat = tf.concat(axis=-1, values=in_features_list)
        return cat

    return concat


def make_add(name):

    def add(in_features1, in_features2):
        with tf.variable_scope(name):
            a = tf.nn.add(in_features1, in_features2)
        return a

    return add


def make_relu(name):

    def relu(in_features):
        with tf.variable_scope(name):
            rl = tf.nn.relu(in_features)
        return rl

    return relu


def make_dropout(keep_prob, name):

    def dropout(in_features):
        with tf.variable_scope(name):
            drop = tf.nn.dropout(in_features, keep_prob=keep_prob)
        return drop

    return dropout


def make_upsample(out_channels, kernel_size, stride, name):
    strides = [1, stride, stride, 1]

    def upsample(in_features, shape):
        in_shape = tf.shape(in_features)
        out_shape = tf.stack([shape[0], shape[1], shape[2], out_channels])

        # Use fixed weights for bilinear upsampling.
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (
            1 - abs(og[1] - center) / factor)

        in_channels = in_features.get_shape()[3].value
        weights = np.zeros(
            (kernel_size, kernel_size, out_channels, in_channels),
            dtype=np.float32)
        weights[:, :, list(range(out_channels)),
                list(range(in_channels))] = filt[..., np.newaxis]
        weights = tf.constant(weights, dtype=tf.float32)
        conv_t = tf.nn.conv2d_transpose(
            in_features, weights, out_shape, strides=strides, padding='SAME')

        return conv_t

    return upsample

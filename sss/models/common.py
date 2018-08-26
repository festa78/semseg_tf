from collections import OrderedDict
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import tensorflow as tf


class Common:
    """This class defines useful common functionals for
    network definitions.
    Assumed to be used as base class.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Used to record size of each layer outputs.
        self.outsizes = OrderedDict()

    def print_outsizes(self):
        self.logger.info('name, size')
        for name, size in self.outsizes.items():
            self.logger.info(name, size)

    def _make_conv_weights(self, shape, std):
        init_op = tf.truncated_normal_initializer(stddev=std)
        return tf.get_variable('weights', shape=shape, initializer=init_op)

    def _make_conv_biases(self, out_channels, constant):
        init_op = tf.constant_initializer(constant)
        return tf.get_variable(
            name='biases', shape=[out_channels], initializer=init_op)

    def _make_conv2d(self, out_channels, kernel_size, stride, bias, name):
        strides = [1, stride, stride, 1]

        def conv2d(in_features):
            with tf.variable_scope(name):
                in_channels = in_features.get_shape()[3].value
                # He initialization.
                std = (2. / in_channels)**.5
                weights_shape = [
                    kernel_size, kernel_size, in_channels, out_channels
                ]
                weights = self._make_conv_weights(weights_shape, std)
                conv = tf.nn.conv2d(
                    in_features, weights, strides, padding='SAME')
                if bias is True:
                    biases = self._make_conv_biases(out_channels, 0.)
                    conv = tf.nn.bias_add(conv, biases)
            self.outsizes[name] = tf.shape(conv)
            return conv

        return conv2d

    def _make_atrous_conv2d(self, out_channels, kernel_size, atrous_rate, bias,
                            name):

        def atrous_conv2d(in_features):
            with tf.variable_scope(name):
                in_channels = in_features.get_shape()[3].value
                # He initialization.
                std = (2. / in_channels)**.5
                weights_shape = [
                    kernel_size, kernel_size, in_channels, out_channels
                ]
                weights = self._make_conv_weights(weights_shape, std)

                conv = tf.nn.atrous_conv2d(
                    in_features, weights, atrous_rate, padding='SAME')
                if bias is True:
                    biases = self._make_conv_biases(out_channels, 0.)
                    conv = tf.nn.bias_add(conv, biases)
            self.outsizes[name] = tf.shape(conv)
            return conv

        return atrous_conv2d

    def _make_avg_pool2d(self, kernel_size, stride, name):
        kernel_sizes = [1, kernel_size, kernel_size, 1]
        strides = [1, stride, stride, 1]

        def avg_pool2d(in_features):
            with tf.variable_scope(name):
                pool = tf.nn.avg_pool(
                    in_features, kernel_sizes, strides, padding='SAME')
            self.outsizes[name] = tf.shape(pool)
            return pool

        return avg_pool2d

    def _make_max_pool2d(self, kernel_size, stride, name):
        kernel_sizes = [1, kernel_size, kernel_size, 1]
        strides = [1, stride, stride, 1]

        def max_pool2d(in_features):
            with tf.variable_scope(name):
                pool = tf.nn.max_pool(
                    in_features, kernel_sizes, strides, padding='SAME')
            self.outsizes[name] = tf.shape(pool)
            return pool

        return max_pool2d

    def _make_bn2d(self, training, name):

        def bn2d(in_features):
            with tf.variable_scope(name):
                bn = tf.layers.batch_normalization(
                    in_features,
                    momentum=0.95,
                    epsilon=1.e-5,
                    training=training)
            self.outsizes[name] = tf.shape(bn)
            return bn

        return bn2d

    def _make_concat(self, name):

        def concat(in_features_list):
            with tf.variable_scope(name):
                cat = tf.concat(axis=-1, values=in_features_list)
            self.outsizes[name] = tf.shape(cat)
            return cat

        return concat

    def _make_add(self, name):

        def add(in_features1, in_features2):
            with tf.variable_scope(name):
                a = tf.add(in_features1, in_features2)
            self.outsizes[name] = tf.shape(a)
            return a

        return add

    def _make_relu(self, name):

        def relu(in_features):
            with tf.variable_scope(name):
                rl = tf.nn.relu(in_features)
            self.outsizes[name] = tf.shape(rl)
            return rl

        return relu

    def _make_dropout(self, keep_prob, name):

        def dropout(in_features):
            with tf.variable_scope(name):
                drop = tf.nn.dropout(in_features, keep_prob=keep_prob)
            self.outsizes[name] = tf.shape(drop)
            return drop

        return dropout

    def _make_upsample(self, out_channels, kernel_size, stride, name):
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
            weights[:, :,
                    list(range(out_channels)),
                    list(range(in_channels))] = filt[..., np.newaxis]
            weights = tf.constant(weights, dtype=tf.float32)
            conv_t = tf.nn.conv2d_transpose(
                in_features,
                weights,
                out_shape,
                strides=strides,
                padding='SAME')

            self.outsizes[name] = tf.shape(conv_t)
            return conv_t

        return upsample

    def _make_resize_bilinear(self, name):

        def resize_bilinear(in_features, size):
            with tf.variable_scope(name):
                rb = tf.image.resize_bilinear(in_features, size=size)
            self.outsizes[name] = tf.shape(rb)
            return rb

        return resize_bilinear

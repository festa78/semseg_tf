from collections import OrderedDict
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from tensorflow.contrib.slim.python.slim.nets import vgg
import tensorflow as tf


class FCN:
    """Fully convolutional network (FCN) implementation.
    cf. https://arxiv.org/abs/1411.4038

    Parameters
    ----------
    num_classes: int
        The number of output classes.
    mode: str
        Which model architecture to use from fcn32, fcn16, and fcn8.
    """
    MODES = ('fcn32', 'fcn16', 'fcn8')

    def __init__(self, num_classes, mode='fcn32'):
        assert mode in self.MODES
        self.logger = logging.getLogger(__name__)
        self.num_classes = num_classes
        self.mode = mode
        self.outsizes = OrderedDict()

        # Prepare layers.
        self.conv1_1 = self._make_conv2d(
            out_channels=64, kernel_size=3, name='conv1/conv1_1')
        self.conv1_2 = self._make_conv2d(
            out_channels=64, kernel_size=3, name='conv1/conv1_2')
        self.pool1 = self._make_max_pool2d(
            kernel_size=2, stride=2, name='pool1')

        self.conv2_1 = self._make_conv2d(
            out_channels=128, kernel_size=3, name='conv2/conv2_1')
        self.conv2_2 = self._make_conv2d(
            out_channels=128, kernel_size=3, name='conv2/conv2_2')
        self.pool2 = self._make_max_pool2d(
            kernel_size=2, stride=2, name='pool2')

        self.conv3_1 = self._make_conv2d(
            out_channels=256, kernel_size=3, name='conv3/conv3_1')
        self.conv3_2 = self._make_conv2d(
            out_channels=256, kernel_size=3, name='conv3/conv3_2')
        self.conv3_3 = self._make_conv2d(
            out_channels=256, kernel_size=3, name='conv3/conv3_3')
        self.pool3 = self._make_max_pool2d(
            kernel_size=2, stride=2, name='pool3')

        self.conv4_1 = self._make_conv2d(
            out_channels=512, kernel_size=3, name='conv4/conv4_1')
        self.conv4_2 = self._make_conv2d(
            out_channels=512, kernel_size=3, name='conv4/conv4_2')
        self.conv4_3 = self._make_conv2d(
            out_channels=512, kernel_size=3, name='conv4/conv4_3')
        self.pool4 = self._make_max_pool2d(
            kernel_size=2, stride=2, name='pool4')

        self.conv5_1 = self._make_conv2d(
            out_channels=512, kernel_size=3, name='conv5/conv5_1')
        self.conv5_2 = self._make_conv2d(
            out_channels=512, kernel_size=3, name='conv5/conv5_2')
        self.conv5_3 = self._make_conv2d(
            out_channels=512, kernel_size=3, name='conv5/conv5_3')
        self.pool5 = self._make_max_pool2d(
            kernel_size=2, stride=2, name='pool5')

        # Use conv2d instead of fc.
        self.fc6 = self._make_conv2d(
            out_channels=4096, kernel_size=7, name='fc6')
        self.dropout6 = self._make_dropout(keep_prob=.5, name='dropout6')

        self.fc7 = self._make_conv2d(
            out_channels=4096, kernel_size=1, name='fc7')
        self.dropout7 = self._make_dropout(keep_prob=.5, name='dropout7')

        self.score_fr = self._make_conv2d(
            out_channels=self.num_classes, kernel_size=1, name='score_fr')

        if self.mode == 'fcn32':
            self.upscore32 = self._make_upscore(
                out_channels=self.num_classes,
                kernel_size=64,
                stride=32,
                name='upscore32')

        if self.mode in ('fcn16', 'fcn8'):
            self.upscore2 = self._make_upscore(
                out_channels=self.num_classes,
                kernel_size=4,
                stride=2,
                name='upscore2')

            self.score_pool4 = self._make_conv2d(
                out_channels=self.num_classes,
                kernel_size=1,
                name='score_pool4')

            if self.mode == 'fcn16':
                self.upscore16 = self._make_upscore(
                    out_channels=self.num_classes,
                    kernel_size=32,
                    stride=16,
                    name='upscore16')
            else:
                self.upscore_pool4 = self._make_upscore(
                    out_channels=self.num_classes,
                    kernel_size=4,
                    stride=2,
                    name='upscore_pool4')

                self.upscore8 = self._make_upscore(
                    out_channels=self.num_classes,
                    kernel_size=16,
                    stride=8,
                    name='upscore8')

                self.score_pool3 = self._make_conv2d(
                    out_channels=self.num_classes,
                    kernel_size=1,
                    name='score_pool3')

    def _make_conv_weights(self, shape, std):
        init_op = tf.truncated_normal_initializer(stddev=std)
        return tf.get_variable('weights', shape=shape, initializer=init_op)

    def _make_conv_biases(self, out_channels, constant):
        init_op = tf.constant_initializer(constant)
        return tf.get_variable(
            name='biases', shape=[out_channels], initializer=init_op)

    def _make_conv2d(self, out_channels, kernel_size, name):

        def conv2d(in_features):
            with tf.variable_scope(name):
                in_channels = in_features.get_shape()[3].value
                # He initialization.
                std = (2. / in_channels)**.5
                weights_shape = [
                    kernel_size, kernel_size, in_channels, out_channels
                ]
                weights = self._make_conv_weights(weights_shape, std)
                biases = self._make_conv_biases(out_channels, 0.)
                conv = tf.nn.conv2d(
                    in_features, weights, [1, 1, 1, 1], padding='SAME')
                conv = tf.nn.bias_add(conv, biases)
            self.outsizes[name] = tf.shape(conv)
            return conv

        return conv2d

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

    def _make_dropout(self, keep_prob, name):

        def dropout(in_features):
            with tf.variable_scope(name):
                drop = tf.nn.dropout(in_features, keep_prob=keep_prob)
            self.outsizes[name] = tf.shape(drop)
            return drop

        return dropout

    def _make_upscore(self, out_channels, kernel_size, stride, name):
        strides = [1, stride, stride, 1]

        def upscore(in_features, shape):
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

        return upscore

    @staticmethod
    def restore_vgg_weights(sess, vgg_pretrain_ckpt_path, scope_prefix='/'):
        """Restore pretrained vgg weights.
        Parameters
        ----------
        sess: tf.Session()
            The current session.
        vgg_pretrain_ckpy_path: str
            The path to the VGG weights ckpt file.
        """
        # Make the dictionary to correspond variables between fcn and vgg.
        var_list = {}
        var_list_fcn = tf.get_collection(
            key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_prefix + 'fcn')
        for var in var_list_fcn:
            if 'score' not in var.name:
                var_list[var.name.replace(scope_prefix + 'fcn',
                                          'vgg_16')[:-2]] = var
        vgg_saver = tf.train.Saver(var_list=var_list)
        vgg_saver.restore(sess, vgg_pretrain_ckpt_path)

    def forward(self, x):
        """Forward the input tensor through the network.
        Parameters
        ----------
        x: (N, H, W, C) tf.Tensor
            Input tensor to process.

        Returns
        -------
        out: (N, H, W, C) tf.Tensor
            The output tensor of the network.
        """
        with tf.variable_scope('fcn', reuse=tf.AUTO_REUSE):
            x_size = tf.shape(x)
            out = self.conv1_1(x)
            out = self.conv1_2(out)
            out = self.pool1(out)

            out = self.conv2_1(out)
            out = self.conv2_2(out)
            out = self.pool2(out)

            out = self.conv3_1(out)
            out = self.conv3_2(out)
            out = self.conv3_3(out)
            out = self.pool3(out)

            if self.mode == 'fcn8':
                pool3 = out

            out = self.conv4_1(out)
            out = self.conv4_2(out)
            out = self.conv4_3(out)
            out = self.pool4(out)

            if self.mode in ('fcn16', 'fcn8'):
                pool4 = out

            out = self.conv5_1(out)
            out = self.conv5_2(out)
            out = self.conv5_3(out)
            out = self.pool5(out)

            out = self.fc6(out)
            out = self.dropout6(out)

            out = self.fc7(out)
            out = self.dropout7(out)

            out = self.score_fr(out)

            if self.mode == 'fcn32':
                out = self.upscore32(out, x_size)
                return out

            if self.mode in ('fcn16', 'fcn8'):
                out2 = self.score_pool4(pool4 * 0.01)
                out = self.upscore2(out, tf.shape(out2))
                out = tf.add(out, out2)
                if self.mode == 'fcn8':
                    out2 = self.score_pool3(pool3 * 0.0001)
                    out = self.upscore_pool4(out, tf.shape(out2))
                    out = tf.add(out, out2)
                    out = self.upscore8(out, x_size)
                    return out

                out = self.upscore16(out, x_size)
                return out


def fcn32(num_classes):
    return FCN(num_classes, 'fcn32')


def fcn16(num_classes):
    return FCN(num_classes, 'fcn16')


def fcn8(num_classes):
    return FCN(num_classes, 'fcn8')

from collections import OrderedDict
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from tensorflow.contrib.slim.python.slim.nets import vgg
import tensorflow as tf

from sss.models.common import Common


class FCN(Common):
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
        super().__init__()

        assert mode in self.MODES
        self.logger = logging.getLogger(__name__)
        self.num_classes = num_classes
        self.mode = mode

        # Prepare layers.
        # TODO: 100 padding.
        self.conv1_1 = self._make_conv2d(
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv1/conv1_1')
        self.relu1_1 = self._make_relu(name='relu1/relu1_1')
        self.conv1_2 = self._make_conv2d(
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv1/conv1_2')
        self.relu1_2 = self._make_relu(name='relu1/relu1_2')
        self.pool1 = self._make_max_pool2d(
            kernel_size=2, stride=2, name='pool1')

        self.conv2_1 = self._make_conv2d(
            out_channels=128,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv2/conv2_1')
        self.relu2_1 = self._make_relu(name='relu2/relu2_1')
        self.conv2_2 = self._make_conv2d(
            out_channels=128,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv2/conv2_2')
        self.relu2_2 = self._make_relu(name='relu2/relu2_2')
        self.pool2 = self._make_max_pool2d(
            kernel_size=2, stride=2, name='pool2')

        self.conv3_1 = self._make_conv2d(
            out_channels=256,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv3/conv3_1')
        self.relu3_1 = self._make_relu(name='relu3/relu3_1')
        self.conv3_2 = self._make_conv2d(
            out_channels=256,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv3/conv3_2')
        self.relu3_2 = self._make_relu(name='relu3/relu3_2')
        self.conv3_3 = self._make_conv2d(
            out_channels=256,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv3/conv3_3')
        self.relu3_3 = self._make_relu(name='relu3/relu3_3')
        self.pool3 = self._make_max_pool2d(
            kernel_size=2, stride=2, name='pool3')

        self.conv4_1 = self._make_conv2d(
            out_channels=512,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv4/conv4_1')
        self.relu4_1 = self._make_relu(name='relu4/relu4_1')
        self.conv4_2 = self._make_conv2d(
            out_channels=512,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv4/conv4_2')
        self.relu4_2 = self._make_relu(name='relu4/relu4_2')
        self.conv4_3 = self._make_conv2d(
            out_channels=512,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv4/conv4_3')
        self.relu4_3 = self._make_relu(name='relu4/relu4_3')
        self.pool4 = self._make_max_pool2d(
            kernel_size=2, stride=2, name='pool4')

        self.conv5_1 = self._make_conv2d(
            out_channels=512,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv5/conv5_1')
        self.relu5_1 = self._make_relu(name='relu5/relu5_1')
        self.conv5_2 = self._make_conv2d(
            out_channels=512,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv5/conv5_2')
        self.relu5_2 = self._make_relu(name='relu5/relu5_2')
        self.conv5_3 = self._make_conv2d(
            out_channels=512,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv5/conv5_3')
        self.relu5_3 = self._make_relu(name='relu5/relu5_3')
        self.pool5 = self._make_max_pool2d(
            kernel_size=2, stride=2, name='pool5')

        # Use conv2d instead of fc.
        self.fc6 = self._make_conv2d(
            out_channels=4096, kernel_size=7, stride=1, bias=True, name='fc6')
        self.relu6 = self._make_relu(name='relu6')
        self.dropout6 = self._make_dropout(keep_prob=.5, name='dropout6')

        self.fc7 = self._make_conv2d(
            out_channels=4096, kernel_size=1, stride=1, bias=True, name='fc7')
        self.relu7 = self._make_relu(name='relu7')
        self.dropout7 = self._make_dropout(keep_prob=.5, name='dropout7')

        self.score_fr = self._make_conv2d(
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            bias=True,
            name='score_fr')

        if self.mode == 'fcn32':
            self.upscore32 = self._make_upsample(
                out_channels=self.num_classes,
                kernel_size=64,
                stride=32,
                name='upscore32')

        if self.mode in ('fcn16', 'fcn8'):
            self.upscore2 = self._make_upsample(
                out_channels=self.num_classes,
                kernel_size=4,
                stride=2,
                name='upscore2')

            self.score_pool4 = self._make_conv2d(
                out_channels=self.num_classes,
                kernel_size=1,
                stride=1,
                bias=True,
                name='score_pool4')

            if self.mode == 'fcn16':
                self.upscore16 = self._make_upsample(
                    out_channels=self.num_classes,
                    kernel_size=32,
                    stride=16,
                    name='upscore16')
            else:
                self.upscore_pool4 = self._make_upsample(
                    out_channels=self.num_classes,
                    kernel_size=4,
                    stride=2,
                    name='upscore_pool4')

                self.upscore8 = self._make_upsample(
                    out_channels=self.num_classes,
                    kernel_size=16,
                    stride=8,
                    name='upscore8')

                self.score_pool3 = self._make_conv2d(
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    bias=True,
                    name='score_pool3')

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
        var_model = tf.get_collection(
            key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_prefix + 'fcn')
        for var in var_model:
            if 'score' not in var.name:
                var_list[var.name.replace(scope_prefix + 'fcn',
                                          'vgg_16')[:-2]] = var
        vgg_saver = tf.train.Saver(var_list=var_list)
        vgg_saver.restore(sess, vgg_pretrain_ckpt_path)

    def __call__(self, x):
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
            out = self.relu1_1(out)
            out = self.conv1_2(out)
            out = self.relu1_2(out)
            out = self.pool1(out)

            out = self.conv2_1(out)
            out = self.relu2_1(out)
            out = self.conv2_2(out)
            out = self.relu2_2(out)
            out = self.pool2(out)

            out = self.conv3_1(out)
            out = self.relu3_1(out)
            out = self.conv3_2(out)
            out = self.relu3_2(out)
            out = self.conv3_3(out)
            out = self.relu3_3(out)
            out = self.pool3(out)

            if self.mode == 'fcn8':
                pool3 = out

            out = self.conv4_1(out)
            out = self.relu4_1(out)
            out = self.conv4_2(out)
            out = self.relu4_2(out)
            out = self.conv4_3(out)
            out = self.relu4_3(out)
            out = self.pool4(out)

            if self.mode in ('fcn16', 'fcn8'):
                pool4 = out

            out = self.conv5_1(out)
            out = self.relu5_1(out)
            out = self.conv5_2(out)
            out = self.relu5_2(out)
            out = self.conv5_3(out)
            out = self.relu5_3(out)
            out = self.pool5(out)

            out = self.fc6(out)
            out = self.relu6(out)
            out = self.dropout6(out)

            out = self.fc7(out)
            out = self.relu7(out)
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

from collections import OrderedDict
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from tensorflow.contrib.slim.python.slim.nets import vgg
import tensorflow as tf

from sss.models.common import Common


class DilationNet(Common):
    """Multi-scale context aggregation by dilated convolutions.
    cf. https://arxiv.org/abs/1511.07122

    Parameters
    ----------
    num_classes: int
        The number of output classes.
    mode: str
        Which model architecture to use from frontend, dilation7, dilation8, and dilation10.
    """
    MODES = ('frontend', 'dilation7', 'dilation8', 'dilation10')

    def __init__(self, num_classes, mode='dilation10'):
        super().__init__()

        assert mode in self.MODES
        self.logger = logging.getLogger(__name__)
        self.num_classes = num_classes
        self.mode = mode

        # Prepare layers.
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

        self.conv5_1 = self._make_atrous_conv2d(
            out_channels=512,
            kernel_size=3,
            atrous_rate=2,
            bias=True,
            name='conv5/conv5_1')
        self.relu5_1 = self._make_relu(name='relu5/relu5_1')
        self.conv5_2 = self._make_atrous_conv2d(
            out_channels=512,
            kernel_size=3,
            atrous_rate=2,
            bias=True,
            name='conv5/conv5_2')
        self.relu5_2 = self._make_relu(name='relu5/relu5_2')
        self.conv5_3 = self._make_atrous_conv2d(
            out_channels=512,
            kernel_size=3,
            atrous_rate=2,
            bias=True,
            name='conv5/conv5_3')
        self.relu5_3 = self._make_relu(name='relu5/relu5_3')

        # Use conv2d instead of fc.
        self.fc6 = self._make_atrous_conv2d(
            out_channels=4096,
            kernel_size=7,
            atrous_rate=4,
            bias=True,
            name='fc6')
        self.relu6 = self._make_relu(name='relu6')
        self.dropout6 = self._make_dropout(keep_prob=.5, name='dropout6')

        self.fc7 = self._make_conv2d(
            out_channels=4096, kernel_size=1, stride=1, bias=True, name='fc7')
        self.relu7 = self._make_relu(name='relu7')
        self.dropout7 = self._make_dropout(keep_prob=.5, name='dropout7')

        self.final = self._make_conv2d(
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            bias=True,
            name='final')

        if self.mode != 'frontend':
            # TODO: needs manual padding?
            self.ctx_conv1_1 = self._make_conv2d(
                out_channels=num_classes,
                kernel_size=3,
                stride=1,
                bias=True,
                name='ctx_conv1/ctx_conv1_1')
            self.ctx_relu1_1 = self._make_relu(name='ctx_relu1/ctx_relu1_1')
            self.ctx_conv1_2 = self._make_conv2d(
                out_channels=num_classes,
                kernel_size=3,
                stride=1,
                bias=True,
                name='ctx_conv1/ctx_conv1_2')
            self.ctx_relu1_2 = self._make_relu(name='ctx_relu1/ctx_relu1_2')

            self.ctx_conv2_1 = self._make_atrous_conv2d(
                out_channels=num_classes,
                kernel_size=3,
                atrous_rate=2,
                bias=True,
                name='ctx_conv2/ctx_conv2_1')
            self.ctx_relu2_1 = self._make_relu(name='ctx_relu1/ctx_relu2_1')

            self.ctx_conv3_1 = self._make_atrous_conv2d(
                out_channels=num_classes,
                kernel_size=3,
                atrous_rate=4,
                bias=True,
                name='ctx_conv3/ctx_conv3_1')
            self.ctx_relu3_1 = self._make_relu(name='ctx_relu1/ctx_relu3_1')

            self.ctx_conv4_1 = self._make_atrous_conv2d(
                out_channels=num_classes,
                kernel_size=3,
                atrous_rate=8,
                bias=True,
                name='ctx_conv4/ctx_conv4_1')
            self.ctx_relu4_1 = self._make_relu(name='ctx_relu1/ctx_relu4_1')

            if self.mode in ('dilation8', 'dilation10'):
                self.ctx_conv5_1 = self._make_atrous_conv2d(
                    out_channels=num_classes,
                    kernel_size=3,
                    atrous_rate=16,
                    bias=True,
                    name='ctx_conv5/ctx_conv5_1')
                self.ctx_relu5_1 = self._make_relu(name='ctx_relu1/ctx_relu5_1')

                if self.mode == 'dilation10':
                    self.ctx_conv6_1 = self._make_atrous_conv2d(
                        out_channels=num_classes,
                        kernel_size=3,
                        atrous_rate=32,
                        bias=True,
                        name='ctx_conv6/ctx_conv6_1')
                    self.ctx_relu6_1 = self._make_relu(
                        name='ctx_relu1/ctx_relu6_1')

                    self.ctx_conv7_1 = self._make_atrous_conv2d(
                        out_channels=num_classes,
                        kernel_size=3,
                        atrous_rate=64,
                        bias=True,
                        name='ctx_conv7/ctx_conv7_1')
                    self.ctx_relu7_1 = self._make_relu(
                        name='ctx_relu1/ctx_relu7_1')

            self.ctx_fc1 = self._make_conv2d(
                out_channels=num_classes,
                kernel_size=3,
                stride=1,
                bias=True,
                name='ctx_fc1')
            self.ctx_fc1_relu = self._make_relu(name='ctx_fc1_relu')
            self.ctx_final = self._make_conv2d(
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                bias=True,
                name='ctx_final')

        # TODO: should not upsample for diltaion7 and dilation8?
        self.ctx_upsample = self._make_resize_bilinear(
            name='ctx_upsample')

    def __call__(self, x):
        """Forward the input tensor through the network.
        Managed by variable_scope to know which model includes
        which variable.
        TODO: make variable_scope shorter but do the same.

        Parameters
        ----------
        x: (N, H, W, C) tf.Tensor
            Input tensor to process.

        Returns
        -------
        out: (N, H, W, C) tf.Tensor
            The output tensor of the network.
        """
        x_size = tf.shape(x)[1:3]
        with tf.variable_scope('dilation10', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('dilation8', reuse=tf.AUTO_REUSE):
                with tf.variable_scope('dilation7', reuse=tf.AUTO_REUSE):
                    with tf.variable_scope('frontend', reuse=tf.AUTO_REUSE):
                        with tf.variable_scope('vgg_16', reuse=tf.AUTO_REUSE):
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

                            out = self.conv4_1(out)
                            out = self.relu4_1(out)
                            out = self.conv4_2(out)
                            out = self.relu4_2(out)
                            out = self.conv4_3(out)
                            out = self.relu4_3(out)

                            out = self.conv5_1(out)
                            out = self.relu5_1(out)
                            out = self.conv5_2(out)
                            out = self.relu5_2(out)
                            out = self.conv5_3(out)
                            out = self.relu5_3(out)

                            out = self.fc6(out)
                            out = self.relu6(out)
                            out = self.dropout6(out)

                            out = self.fc7(out)
                            out = self.relu7(out)
                            out = self.dropout7(out)
                        # vgg_16

                        out = self.final(out)
                        if self.mode == 'frontend':
                            out = self.ctx_upsample(out, x_size)
                            return out
                    # frontend

                    out = self.ctx_conv1_1(out)
                    out = self.ctx_relu1_1(out)
                    out = self.ctx_conv1_2(out)
                    out = self.ctx_relu1_2(out)

                    out = self.ctx_conv2_1(out)
                    out = self.ctx_relu2_1(out)

                    out = self.ctx_conv3_1(out)
                    out = self.ctx_relu3_1(out)

                    out = self.ctx_conv4_1(out)
                    out = self.ctx_relu4_1(out)
                    if self.mode == 'dilation7':
                        out = self.ctx_upsample(out, x_size)
                        return out
                # dilation7

                out = self.ctx_conv5_1(out)
                out = self.ctx_relu5_1(out)
                if self.mode == 'dilation8':
                    out = self.ctx_upsample(out, x_size)
                    return out
            # dilation8

            out = self.ctx_conv6_1(out)
            out = self.ctx_relu6_1(out)

            out = self.ctx_conv7_1(out)
            out = self.ctx_relu7_1(out)

            out = self.ctx_fc1(out)
            out = self.ctx_fc1_relu(out)

            out = self.ctx_final(out)

            out = self.ctx_upsample(out, x_size)
        # dilation10

        return out


def frontend(num_classes):
    return DilationNet(num_classes, 'frontend')


def dilation7(num_classes):
    return DilationNet(num_classes, 'dilation7')


def dilation8(num_classes):
    return DilationNet(num_classes, 'dilation8')


def dilation10(num_classes):
    return DilationNet(num_classes, 'dilation10')

from collections import OrderedDict
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import tensorflow as tf

from sss.models.common import Common


class PSPNet(Common):
    """Pyramid Scene Parsing Network.
    cf. https://arxiv.org/abs/1612.01105

    Parameters
    ----------
    num_classes: int
        The number of output classes.
    mode: str
        Which model architecture to use from pspnet50 and pspnet101.
    """
    MODES = ('pspnet50', 'pspnet101')

    def __init__(self, num_classes, mode='pspnet101'):
        super().__init__()

        assert mode in self.MODES
        self.logger = logging.getLogger(__name__)
        self.num_classes = num_classes
        self.mode = mode
        # Necessary for batch normalization.
        self.training = True

        # Prepare layers.
        self.conv1_1_3x3_s2 = self._make_conv2d(
            out_channels=64,
            kernel_size=3,
            stride=2,
            bias=False,
            name='conv1_1_3x3_s2')
        self.conv1_1_3x3_s2_bn = self._make_bn2d(
            training=self.training, name='conv1_1_3x3_s2/bn')
        self.conv1_1_3x3_s2_relu = self._make_relu(name='conv1_1_3x3_s2/relu')
        self.conv1_2_3x3 = self._make_conv2d(
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=False,
            name='conv1_2_3x3')
        self.conv1_2_3x3_bn = self._make_bn2d(
            training=self.training, name='conv1_2_3x3_bn')
        self.conv1_2_3x3_relu = self._make_relu(name='conv1_2_3x3/relu')
        self.conv1_3_3x3 = self._make_conv2d(
            out_channels=128,
            kernel_size=3,
            stride=1,
            bias=False,
            name='conv1_3_3x3')
        self.conv1_3_3x3_bn = self._make_bn2d(
            training=self.training, name='conv1_3_3x3_bn')
        self.conv1_3_3x3_relu = self._make_relu(name='conv1_3_3x3/relu')
        self.pool1_3x3_s2 = self._make_max_pool2d(
            kernel_size=3, stride=2, name='pool1_3x3_s2')
        self.conv2_1_1x1_proj = self._make_conv2d(
            out_channels=256,
            kernel_size=1,
            stride=1,
            bias=False,
            name='conv2_1_1x1_proj')
        self.conv2_1_1x1_proj_bn = self._make_bn2d(
            training=self.training, name='conv2_1_1x1_proj/bn')

        self.conv2_1_block = self._make_block(
            out_channels=[64, 64, 256],
            kernel_sizes=[1, 3, 1],
            strides=[1, 1, 1],
            use_atrous=False,
            name='conv2_1')

        self.conv2_1 = self._make_add(name='conv2_1')
        self.conv2_1_relu = self._make_relu(name='conv2_1/relu')

        self.conv2_2_block = self._make_block(
            out_channels=[64, 64, 256],
            kernel_sizes=[1, 3, 1],
            strides=[1, 1, 1],
            use_atrous=False,
            name='conv2_2')

        self.conv2_2 = self._make_add(name='conv2_2')
        self.conv2_2_relu = self._make_relu(name='conv2_2/relu')

        self.conv2_3_block = self._make_block(
            out_channels=[64, 64, 256],
            kernel_sizes=[1, 3, 1],
            strides=[1, 1, 1],
            use_atrous=False,
            name='conv2_3')

        self.conv2_3 = self._make_add(name='conv2_3')
        self.conv2_3_relu = self._make_relu(name='conv2_3/relu')

        self.conv3_1_1x1_proj = self._make_conv2d(
            out_channels=512,
            kernel_size=1,
            stride=2,
            bias=False,
            name='conv3_1_1x1_proj')
        self.conv3_1_1x1_proj_bn = self._make_bn2d(
            training=self.training, name='conv3_1_1x1_proj/bn')

        # Use different stride.
        if self.mode == 'pspnet50':
            self.conv3_1_block = self._make_block(
                out_channels=[128, 128, 512],
                kernel_sizes=[1, 3, 1],
                strides=[2, 1, 1],
                use_atrous=False,
                name='conv3_1')
        else:
            self.conv3_1_block = self._make_block(
                out_channels=[128, 128, 512],
                kernel_sizes=[1, 3, 1],
                strides=[1, 2, 1],
                use_atrous=False,
                name='conv3_1')

        self.conv3_1 = self._make_add(name='conv3_1')
        self.conv3_1_relu = self._make_relu(name='conv3_1/relu')

        self.conv3_2_block = self._make_block(
            out_channels=[128, 128, 512],
            kernel_sizes=[1, 3, 1],
            strides=[1, 1, 1],
            use_atrous=False,
            name='conv3_2')

        self.conv3_2 = self._make_add(name='conv3_2')
        self.conv3_2_relu = self._make_relu(name='conv3_2/relu')

        self.conv3_3_block = self._make_block(
            out_channels=[128, 128, 512],
            kernel_sizes=[1, 3, 1],
            strides=[1, 1, 1],
            use_atrous=False,
            name='conv3_3')

        self.conv3_3 = self._make_add(name='conv3_3')
        self.conv3_3_relu = self._make_relu(name='conv3_3/relu')

        self.conv3_4_block = self._make_block(
            out_channels=[128, 128, 512],
            kernel_sizes=[1, 3, 1],
            strides=[1, 1, 1],
            use_atrous=False,
            name='conv3_4')

        self.conv3_4 = self._make_add(name='conv3_4')
        self.conv3_4_relu = self._make_relu(name='conv3_4/relu')

        self.conv4_1_1x1_proj = self._make_conv2d(
            out_channels=1024,
            kernel_size=1,
            stride=1,
            bias=False,
            name='conv4_1_1x1_proj')
        self.conv4_1_1x1_proj_bn = self._make_bn2d(
            training=self.training, name='conv4_1_1x1_proj/bn')

        self.conv4_1_block = self._make_block(
            out_channels=[256, 256, 1024],
            kernel_sizes=[1, 3, 1],
            strides=[1, 2, 1],
            use_atrous=True,
            name='conv4_1')

        self.conv4_1 = self._make_add(name='conv4_1')
        self.conv4_1_relu = self._make_relu(name='conv4_1/relu')

        self.conv4_2_block = self._make_block(
            out_channels=[256, 256, 1024],
            kernel_sizes=[1, 3, 1],
            strides=[1, 2, 1],
            use_atrous=True,
            name='conv4_2')

        self.conv4_2 = self._make_add(name='conv4_2')
        self.conv4_2_relu = self._make_relu(name='conv4_2/relu')

        self.conv4_3_block = self._make_block(
            out_channels=[256, 256, 1024],
            kernel_sizes=[1, 3, 1],
            strides=[1, 2, 1],
            use_atrous=True,
            name='conv4_3')

        self.conv4_3 = self._make_add(name='conv4_3')
        self.conv4_3_relu = self._make_relu(name='conv4_3/relu')

        self.conv4_4_block = self._make_block(
            out_channels=[256, 256, 1024],
            kernel_sizes=[1, 3, 1],
            strides=[1, 2, 1],
            use_atrous=True,
            name='conv4_4')

        self.conv4_4 = self._make_add(name='conv4_4')
        self.conv4_4_relu = self._make_relu(name='conv4_4/relu')

        self.conv4_5_block = self._make_block(
            out_channels=[256, 256, 1024],
            kernel_sizes=[1, 3, 1],
            strides=[1, 2, 1],
            use_atrous=True,
            name='conv4_5')

        self.conv4_5 = self._make_add(name='conv4_5')
        self.conv4_5_relu = self._make_relu(name='conv4_5/relu')

        self.conv4_6_block = self._make_block(
            out_channels=[256, 256, 1024],
            kernel_sizes=[1, 3, 1],
            strides=[1, 2, 1],
            use_atrous=True,
            name='conv4_6')

        self.conv4_6 = self._make_add(name='conv4_6')
        self.conv4_6_relu = self._make_relu(name='conv4_6/relu')

        if self.mode == 'pspnet101':
            self.conv4_7_block = self._make_block(
                out_channels=[256, 256, 1024],
                kernel_sizes=[1, 3, 1],
                strides=[1, 2, 1],
                use_atrous=True,
                name='conv4_7')

            self.conv4_7 = self._make_add(name='conv4_7')
            self.conv4_7_relu = self._make_relu(name='conv4_7/relu')

            self.conv4_8_block = self._make_block(
                out_channels=[256, 256, 1024],
                kernel_sizes=[1, 3, 1],
                strides=[1, 2, 1],
                use_atrous=True,
                name='conv4_8')

            self.conv4_8 = self._make_add(name='conv4_8')
            self.conv4_8_relu = self._make_relu(name='conv4_8/relu')

            self.conv4_9_block = self._make_block(
                out_channels=[256, 256, 1024],
                kernel_sizes=[1, 3, 1],
                strides=[1, 2, 1],
                use_atrous=True,
                name='conv4_9')

            self.conv4_9 = self._make_add(name='conv4_9')
            self.conv4_9_relu = self._make_relu(name='conv4_9/relu')

            self.conv4_10_block = self._make_block(
                out_channels=[256, 256, 1024],
                kernel_sizes=[1, 3, 1],
                strides=[1, 2, 1],
                use_atrous=True,
                name='conv4_10')

            self.conv4_10 = self._make_add(name='conv4_10')
            self.conv4_10_relu = self._make_relu(name='conv4_10/relu')

            self.conv4_11_block = self._make_block(
                out_channels=[256, 256, 1024],
                kernel_sizes=[1, 3, 1],
                strides=[1, 2, 1],
                use_atrous=True,
                name='conv4_11')

            self.conv4_11 = self._make_add(name='conv4_11')
            self.conv4_11_relu = self._make_relu(name='conv4_11/relu')

            self.conv4_12_block = self._make_block(
                out_channels=[256, 256, 1024],
                kernel_sizes=[1, 3, 1],
                strides=[1, 2, 1],
                use_atrous=True,
                name='conv4_12')

            self.conv4_12 = self._make_add(name='conv4_12')
            self.conv4_12_relu = self._make_relu(name='conv4_12/relu')

            self.conv4_13_block = self._make_block(
                out_channels=[256, 256, 1024],
                kernel_sizes=[1, 3, 1],
                strides=[1, 2, 1],
                use_atrous=True,
                name='conv4_13')

            self.conv4_13 = self._make_add(name='conv4_13')
            self.conv4_13_relu = self._make_relu(name='conv4_13/relu')

            self.conv4_14_block = self._make_block(
                out_channels=[256, 256, 1024],
                kernel_sizes=[1, 3, 1],
                strides=[1, 2, 1],
                use_atrous=True,
                name='conv4_14')

            self.conv4_14 = self._make_add(name='conv4_14')
            self.conv4_14_relu = self._make_relu(name='conv4_14/relu')

            self.conv4_15_block = self._make_block(
                out_channels=[256, 256, 1024],
                kernel_sizes=[1, 3, 1],
                strides=[1, 2, 1],
                use_atrous=True,
                name='conv4_15')

            self.conv4_15 = self._make_add(name='conv4_15')
            self.conv4_15_relu = self._make_relu(name='conv4_15/relu')

            self.conv4_16_block = self._make_block(
                out_channels=[256, 256, 1024],
                kernel_sizes=[1, 3, 1],
                strides=[1, 2, 1],
                use_atrous=True,
                name='conv4_16')

            self.conv4_16 = self._make_add(name='conv4_16')
            self.conv4_16_relu = self._make_relu(name='conv4_16/relu')

            self.conv4_17_block = self._make_block(
                out_channels=[256, 256, 1024],
                kernel_sizes=[1, 3, 1],
                strides=[1, 2, 1],
                use_atrous=True,
                name='conv4_17')

            self.conv4_17 = self._make_add(name='conv4_17')
            self.conv4_17_relu = self._make_relu(name='conv4_17/relu')

            self.conv4_18_block = self._make_block(
                out_channels=[256, 256, 1024],
                kernel_sizes=[1, 3, 1],
                strides=[1, 2, 1],
                use_atrous=True,
                name='conv4_18')

            self.conv4_18 = self._make_add(name='conv4_18')
            self.conv4_18_relu = self._make_relu(name='conv4_18/relu')

            self.conv4_19_block = self._make_block(
                out_channels=[256, 256, 1024],
                kernel_sizes=[1, 3, 1],
                strides=[1, 2, 1],
                use_atrous=True,
                name='conv4_19')

            self.conv4_19 = self._make_add(name='conv4_19')
            self.conv4_19_relu = self._make_relu(name='conv4_19/relu')

            self.conv4_20_block = self._make_block(
                out_channels=[256, 256, 1024],
                kernel_sizes=[1, 3, 1],
                strides=[1, 2, 1],
                use_atrous=True,
                name='conv4_20')

            self.conv4_20 = self._make_add(name='conv4_20')
            self.conv4_20_relu = self._make_relu(name='conv4_20/relu')

            self.conv4_21_block = self._make_block(
                out_channels=[256, 256, 1024],
                kernel_sizes=[1, 3, 1],
                strides=[1, 2, 1],
                use_atrous=True,
                name='conv4_21')

            self.conv4_21 = self._make_add(name='conv4_21')
            self.conv4_21_relu = self._make_relu(name='conv4_21/relu')

            self.conv4_22_block = self._make_block(
                out_channels=[256, 256, 1024],
                kernel_sizes=[1, 3, 1],
                strides=[1, 2, 1],
                use_atrous=True,
                name='conv4_22')

            self.conv4_22 = self._make_add(name='conv4_22')
            self.conv4_22_relu = self._make_relu(name='conv4_22/relu')

            self.conv4_23_block = self._make_block(
                out_channels=[256, 256, 1024],
                kernel_sizes=[1, 3, 1],
                strides=[1, 2, 1],
                use_atrous=True,
                name='conv4_23')

            self.conv4_23 = self._make_add(name='conv4_23')
            self.conv4_23_relu = self._make_relu(name='conv4_23/relu')
        # if self.mode == 'pspnet101' ends.

        self.conv5_1_1x1_proj = self._make_conv2d(
            out_channels=2048,
            kernel_size=1,
            stride=1,
            bias=False,
            name='conv5_1_1x1_proj')
        self.conv5_1_1x1_proj_bn = self._make_bn2d(
            training=self.training, name='conv5_1_1x1_proj/bn')

        self.conv5_1_block = self._make_block(
            out_channels=[512, 512, 2048],
            kernel_sizes=[1, 3, 1],
            strides=[1, 4, 1],
            use_atrous=True,
            name='conv5_1')

        self.conv5_1 = self._make_add(name='conv5_1')
        self.conv5_1_relu = self._make_relu(name='conv5_1/relu')

        self.conv5_2_block = self._make_block(
            out_channels=[512, 512, 2048],
            kernel_sizes=[1, 3, 1],
            strides=[1, 4, 1],
            use_atrous=True,
            name='conv5_2')

        self.conv5_2 = self._make_add(name='conv5_2')
        self.conv5_2_relu = self._make_relu(name='conv5_2/relu')

        self.conv5_3_block = self._make_block(
            out_channels=[512, 512, 2048],
            kernel_sizes=[1, 3, 1],
            strides=[1, 4, 1],
            use_atrous=True,
            name='conv5_3')

        self.conv5_3 = self._make_add(name='conv5_3')
        self.conv5_3_relu = self._make_relu(name='conv5_3/relu')

        if self.mode == 'pspnet50':
            self.conv5_3_pool1 = self._make_block2(
                kernel_size=60, stride=60, name='conv5_3_pool1')
            self.conv5_3_pool2 = self._make_block2(
                kernel_size=30, stride=30, name='conv5_3_pool2')
            self.conv5_3_pool3 = self._make_block2(
                kernel_size=20, stride=20, name='conv5_3_pool3')
            self.conv5_3_pool6 = self._make_block2(
                kernel_size=10, stride=10, name='conv5_3_pool6')
        else:
            self.conv5_3_pool1 = self._make_block2(
                kernel_size=90, stride=90, name='conv5_3_pool1')
            self.conv5_3_pool2 = self._make_block2(
                kernel_size=45, stride=45, name='conv5_3_pool2')
            self.conv5_3_pool3 = self._make_block2(
                kernel_size=30, stride=30, name='conv5_3_pool3')
            self.conv5_3_pool6 = self._make_block2(
                kernel_size=15, stride=15, name='conv5_3_pool6')

        self.conv5_3_concat = self._make_concat(name='conv5_3_concat')

        self.conv5_4 = self._make_conv2d(
            out_channels=512,
            kernel_size=3,
            stride=1,
            bias=False,
            name='conv5_4')
        self.conv5_4_bn = self._make_bn2d(
            training=self.training, name='conv5_4/bn')
        self.conv5_4_relu = self._make_relu(name='conv5_4/relu')
        self.conv5_4_dropout = self._make_dropout(
            keep_prob=0.1, name='conv5_4/dropout')

        self.conv6 = self._make_conv2d(
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            bias=True,
            name='conv6')

        self.conv6_interp = self._make_resize_bilinear(name='conv6/interp')

    def _make_block2(self, kernel_size, stride, name):
        conv_pool = self._make_avg_pool2d(
            kernel_size=kernel_size, stride=stride, name=name)
        conv_pool_conv = self._make_conv2d(
            out_channels=512,
            kernel_size=1,
            stride=1,
            bias=False,
            name=name + '/conv')
        conv_pool_bn = self._make_bn2d(
            training=self.training, name=name + '/bn')
        conv_pool_relu = self._make_relu(name=name + '/relu')
        conv_pool_interp = self._make_resize_bilinear(name=name + '/interp')

        def block2(in_features, size):
            out = conv_pool(in_features)
            out = conv_pool_conv(out)
            out = conv_pool_bn(out)
            out = conv_pool_relu(out)
            out = conv_pool_interp(out, size)
            return out

        return block2

    def _make_block(self, out_channels, kernel_sizes, strides, use_atrous,
                    name):
        assert len(out_channels) == 3
        assert len(kernel_sizes) == 3
        assert len(strides) == 3

        # TODO: Needs manual padding?
        conv_1x1_reduce = self._make_conv2d(
            out_channels=out_channels[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            bias=False,
            name=name + '_1x1_reduce')
        conv_1x1_reduce_bn = self._make_bn2d(
            training=self.training, name=name + '_1x1_reduce/bn')
        conv_1x1_reduce_relu = self._make_relu(name=name + '_1x1_reduce/relu')
        if use_atrous == True:
            conv_3x3 = self._make_atrous_conv2d(
                out_channels=out_channels[1],
                kernel_size=kernel_sizes[1],
                atrous_rate=strides[1],
                bias=False,
                name=name + '_3x3')
        else:
            conv_3x3 = self._make_conv2d(
                out_channels=out_channels[1],
                kernel_size=kernel_sizes[1],
                stride=strides[1],
                bias=False,
                name=name + '_3x3')
        conv_3x3_bn = self._make_bn2d(
            training=self.training, name=name + '_3x3/bn')
        conv_3x3_relu = self._make_relu(name=name + '_3x3/relu')
        conv_1x1_increase = self._make_conv2d(
            out_channels=out_channels[2],
            kernel_size=kernel_sizes[2],
            stride=strides[2],
            bias=False,
            name=name + '_1x1_increase')
        conv_1x1_increase_bn = self._make_bn2d(
            training=self.training, name=name + '_1x1_increase/bn')

        def block(in_features):
            out = conv_1x1_reduce(in_features)
            out = conv_1x1_reduce_bn(out)
            out = conv_1x1_reduce_relu(out)
            out = conv_3x3(out)
            out = conv_3x3_bn(out)
            out = conv_3x3_relu(out)
            out = conv_1x1_increase(out)
            out = conv_1x1_increase_bn(out)
            return out

        return block

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
        x_size = tf.shape(x)[1:3]
        with tf.variable_scope('pspnet101', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('pspnet50', reuse=tf.AUTO_REUSE):
                out = self.conv1_1_3x3_s2(x)
                out = self.conv1_1_3x3_s2_bn(out)
                out = self.conv1_1_3x3_s2_relu(out)
                out = self.conv1_2_3x3(out)
                out = self.conv1_2_3x3_bn(out)
                out = self.conv1_2_3x3_relu(out)
                out = self.conv1_3_3x3(out)
                out = self.conv1_3_3x3_bn(out)
                out = self.conv1_3_3x3_relu(out)
                out2 = self.pool1_3x3_s2(out)

                out = self.conv2_1_1x1_proj(out2)
                out = self.conv2_1_1x1_proj_bn(out)

                out2 = self.conv2_1_block(out2)
                out = self.conv2_1(out, out2)
                out = self.conv2_1_relu(out)

                out2 = self.conv2_2_block(out)
                out = self.conv2_2(out, out2)
                out = self.conv2_2_relu(out)

                out2 = self.conv2_3_block(out)
                out = self.conv2_3(out, out2)
                out2 = self.conv2_3_relu(out)

                out = self.conv3_1_1x1_proj(out2)
                out = self.conv3_1_1x1_proj_bn(out)

                out2 = self.conv3_1_block(out2)
                out = self.conv3_1(out, out2)
                out = self.conv3_1_relu(out)

                out2 = self.conv3_2_block(out)
                out = self.conv3_2(out, out2)
                out = self.conv3_2_relu(out)

                out2 = self.conv3_3_block(out)
                out = self.conv3_3(out, out2)
                out = self.conv3_3_relu(out)

                out2 = self.conv3_4_block(out)
                out = self.conv3_4(out, out2)
                out2 = self.conv3_4_relu(out)

                out = self.conv4_1_1x1_proj(out2)
                out = self.conv4_1_1x1_proj_bn(out)

                out2 = self.conv4_1_block(out2)
                out = self.conv4_1(out, out2)
                out = self.conv4_1_relu(out)

                out2 = self.conv4_2_block(out)
                out = self.conv4_2(out, out2)
                out = self.conv4_2_relu(out)

                out2 = self.conv4_3_block(out)
                out = self.conv4_3(out, out2)
                out = self.conv4_3_relu(out)

                out2 = self.conv4_4_block(out)
                out = self.conv4_4(out, out2)
                out = self.conv4_4_relu(out)

                out2 = self.conv4_5_block(out)
                out = self.conv4_5(out, out2)
                out = self.conv4_5_relu(out)

                out2 = self.conv4_6_block(out)
                out = self.conv4_6(out, out2)
                out = self.conv4_6_relu(out)
            # pspnet50

            if self.mode == 'pspnet101':
                out2 = self.conv4_7_block(out)
                out = self.conv4_7(out, out2)
                out = self.conv4_7_relu(out)

                out2 = self.conv4_8_block(out)
                out = self.conv4_8(out, out2)
                out = self.conv4_8_relu(out)

                out2 = self.conv4_9_block(out)
                out = self.conv4_9(out, out2)
                out = self.conv4_9_relu(out)

                out2 = self.conv4_10_block(out)
                out = self.conv4_10(out, out2)
                out = self.conv4_10_relu(out)

                out2 = self.conv4_11_block(out)
                out = self.conv4_11(out, out2)
                out = self.conv4_11_relu(out)

                out2 = self.conv4_12_block(out)
                out = self.conv4_12(out, out2)
                out = self.conv4_12_relu(out)

                out2 = self.conv4_13_block(out)
                out = self.conv4_13(out, out2)
                out = self.conv4_13_relu(out)

                out2 = self.conv4_14_block(out)
                out = self.conv4_14(out, out2)
                out = self.conv4_14_relu(out)

                out2 = self.conv4_15_block(out)
                out = self.conv4_15(out, out2)
                out = self.conv4_15_relu(out)

                out2 = self.conv4_16_block(out)
                out = self.conv4_16(out, out2)
                out = self.conv4_16_relu(out)

                out2 = self.conv4_17_block(out)
                out = self.conv4_17(out, out2)
                out = self.conv4_17_relu(out)

                out2 = self.conv4_18_block(out)
                out = self.conv4_18(out, out2)
                out = self.conv4_18_relu(out)

                out2 = self.conv4_19_block(out)
                out = self.conv4_19(out, out2)
                out = self.conv4_19_relu(out)

                out2 = self.conv4_20_block(out)
                out = self.conv4_20(out, out2)
                out = self.conv4_20_relu(out)

                out2 = self.conv4_21_block(out)
                out = self.conv4_21(out, out2)
                out = self.conv4_21_relu(out)

                out2 = self.conv4_22_block(out)
                out = self.conv4_22(out, out2)
                out = self.conv4_22_relu(out)

                out2 = self.conv4_23_block(out)
                out = self.conv4_23(out, out2)
                out2 = self.conv4_23_relu(out)
            # if self.mode == 'pspnet101' ends.

            with tf.variable_scope('pspnet50', reuse=tf.AUTO_REUSE):
                out = self.conv5_1_1x1_proj(out2)
                out = self.conv5_1_1x1_proj_bn(out)

                out2 = self.conv5_1_block(out2)
                out = self.conv5_1(out, out2)
                out = self.conv5_1_relu(out)

                out2 = self.conv5_2_block(out)
                out = self.conv5_2(out, out2)
                out = self.conv5_2_relu(out)

                out2 = self.conv5_3_block(out)
                out = self.conv5_3(out, out2)
                out = self.conv5_3_relu(out)

                out_size = tf.shape(out)[1:3]

                out2 = self.conv5_3_pool6(out, out_size)
                out3 = self.conv5_3_pool3(out, out_size)
                out4 = self.conv5_3_pool2(out, out_size)
                out5 = self.conv5_3_pool1(out, out_size)
                out = self.conv5_3_concat([out, out2, out3, out4, out5])

                out = self.conv5_4(out)
                out = self.conv5_4_bn(out)
                out = self.conv5_4_relu(out)
                out = self.conv5_4_dropout(out)

                out = self.conv6(out)
                out = self.conv6_interp(out, x_size)
            # pspnet50
        # pspnet101

        return out


def pspnet50(num_classes):
    return PSPNet(num_classes, 'pspnet50')


def pspnet101(num_classes):
    return PSPNet(num_classes, 'pspnet101')

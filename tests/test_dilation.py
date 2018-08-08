"""Test set for FCN classes.
"""

import numpy as np
import tensorflow as tf

import project_root

from sss.models.dilation_net import dilation7, dilation8, dilation10
from sss.utils.losses import cross_entropy


def test_dilation_init():
    """Test DilationNet class initialization.
    """
    NUM_CLASSES = 5
    CKPT_PATH = 'dummy'

    with tf.Graph().as_default():
        # Check methods constructed in __init__().
        dut = dilation7(NUM_CLASSES)
        assert dut.logger is not None
        assert dut.num_classes == NUM_CLASSES
        assert hasattr(dut, 'conv1_1')
        assert hasattr(dut, 'relu1_1')
        assert hasattr(dut, 'conv1_2')
        assert hasattr(dut, 'relu1_2')
        assert hasattr(dut, 'pool1')
        assert hasattr(dut, 'conv2_1')
        assert hasattr(dut, 'relu2_1')
        assert hasattr(dut, 'conv2_2')
        assert hasattr(dut, 'relu2_2')
        assert hasattr(dut, 'pool2')
        assert hasattr(dut, 'conv3_1')
        assert hasattr(dut, 'relu3_1')
        assert hasattr(dut, 'conv3_2')
        assert hasattr(dut, 'relu3_2')
        assert hasattr(dut, 'conv3_3')
        assert hasattr(dut, 'relu3_3')
        assert hasattr(dut, 'pool3')
        assert hasattr(dut, 'conv4_1')
        assert hasattr(dut, 'relu4_1')
        assert hasattr(dut, 'conv4_2')
        assert hasattr(dut, 'relu4_2')
        assert hasattr(dut, 'conv4_3')
        assert hasattr(dut, 'relu4_3')
        assert hasattr(dut, 'conv5_1')
        assert hasattr(dut, 'relu5_1')
        assert hasattr(dut, 'conv5_2')
        assert hasattr(dut, 'relu5_2')
        assert hasattr(dut, 'conv5_3')
        assert hasattr(dut, 'relu5_3')
        assert hasattr(dut, 'fc6')
        assert hasattr(dut, 'relu6')
        assert hasattr(dut, 'dropout6')
        assert hasattr(dut, 'fc7')
        assert hasattr(dut, 'relu7')
        assert hasattr(dut, 'dropout7')
        assert hasattr(dut, 'final')
        assert hasattr(dut, 'ctx_conv1_1')
        assert hasattr(dut, 'ctx_relu1_1')
        assert hasattr(dut, 'ctx_conv1_2')
        assert hasattr(dut, 'ctx_relu1_2')
        assert hasattr(dut, 'ctx_conv2_1')
        assert hasattr(dut, 'ctx_relu2_1')
        assert hasattr(dut, 'ctx_conv3_1')
        assert hasattr(dut, 'ctx_relu3_1')
        assert hasattr(dut, 'ctx_conv4_1')
        assert hasattr(dut, 'ctx_relu4_1')
        assert hasattr(dut, 'ctx_fc1')
        assert hasattr(dut, 'ctx_fc1_relu')
        assert hasattr(dut, 'ctx_final')

        dut = dilation8(NUM_CLASSES)
        assert hasattr(dut, 'ctx_conv5_1')
        assert hasattr(dut, 'ctx_relu5_1')

        dut = dilation10(NUM_CLASSES)
        assert hasattr(dut, 'ctx_conv6_1')
        assert hasattr(dut, 'ctx_relu6_1')
        assert hasattr(dut, 'ctx_conv7_1')
        assert hasattr(dut, 'ctx_relu7_1')
        assert hasattr(dut, 'ctx_upsample')


def test_dilation_architecture():
    """Test the architecture of DilationNet.
    """
    IMAGE_SIZE = 224
    NUM_CLASSES = 5
    GT_VAR_LIST = sorted([
        "dilation/conv1/conv1_1/weights:0",
        "dilation/conv1/conv1_1/biases:0",
        "dilation/conv1/conv1_2/weights:0",
        "dilation/conv1/conv1_2/biases:0",
        "dilation/conv2/conv2_1/weights:0",
        "dilation/conv2/conv2_1/biases:0",
        "dilation/conv2/conv2_2/weights:0",
        "dilation/conv2/conv2_2/biases:0",
        "dilation/conv3/conv3_1/weights:0",
        "dilation/conv3/conv3_1/biases:0",
        "dilation/conv3/conv3_2/weights:0",
        "dilation/conv3/conv3_2/biases:0",
        "dilation/conv3/conv3_3/weights:0",
        "dilation/conv3/conv3_3/biases:0",
        "dilation/conv4/conv4_1/weights:0",
        "dilation/conv4/conv4_1/biases:0",
        "dilation/conv4/conv4_2/weights:0",
        "dilation/conv4/conv4_2/biases:0",
        "dilation/conv4/conv4_3/weights:0",
        "dilation/conv4/conv4_3/biases:0",
        "dilation/conv5/conv5_1/weights:0",
        "dilation/conv5/conv5_1/biases:0",
        "dilation/conv5/conv5_2/weights:0",
        "dilation/conv5/conv5_2/biases:0",
        "dilation/conv5/conv5_3/weights:0",
        "dilation/conv5/conv5_3/biases:0",
        "dilation/fc6/weights:0",
        "dilation/fc6/biases:0",
        "dilation/fc7/weights:0",
        "dilation/fc7/biases:0",
        "dilation/final/weights:0",
        "dilation/final/biases:0",
        "dilation/ctx_conv1/ctx_conv1_1/weights:0",
        "dilation/ctx_conv1/ctx_conv1_1/biases:0",
        "dilation/ctx_conv1/ctx_conv1_2/weights:0",
        "dilation/ctx_conv1/ctx_conv1_2/biases:0",
        "dilation/ctx_conv2/ctx_conv2_1/weights:0",
        "dilation/ctx_conv2/ctx_conv2_1/biases:0",
        "dilation/ctx_conv3/ctx_conv3_1/weights:0",
        "dilation/ctx_conv3/ctx_conv3_1/biases:0",
        "dilation/ctx_conv4/ctx_conv4_1/weights:0",
        "dilation/ctx_conv4/ctx_conv4_1/biases:0",
        "dilation/ctx_conv5/ctx_conv5_1/weights:0",
        "dilation/ctx_conv5/ctx_conv5_1/biases:0",
        "dilation/ctx_conv6/ctx_conv6_1/weights:0",
        "dilation/ctx_conv6/ctx_conv6_1/biases:0",
        "dilation/ctx_conv7/ctx_conv7_1/weights:0",
        "dilation/ctx_conv7/ctx_conv7_1/biases:0",
        "dilation/ctx_fc1/weights:0",
        "dilation/ctx_fc1/biases:0",
        "dilation/ctx_final/weights:0",
        "dilation/ctx_final/biases:0",
    ])

    with tf.Graph().as_default():
        dut = dilation10(NUM_CLASSES)
        dummy_in = tf.placeholder(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE, 3))
        dut.forward(dummy_in)
        var_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES)
        var_list = sorted(var.name for var in var_list)
        assert var_list == GT_VAR_LIST


def test_dilation_update():
    """Test DilationNet surely updates the parameters.
    cf. https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
    """
    IMAGE_SIZE = 224
    NUM_CLASSES = 5
    MODELS = (dilation7, dilation8, dilation10)

    for model in MODELS:
        with tf.Graph().as_default():
            with tf.device("/cpu:0"):
                dummy_in = tf.placeholder(tf.float32,
                                          (None, IMAGE_SIZE, IMAGE_SIZE, 3))
                dummy_gt = tf.placeholder(
                    tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1])

            with tf.device("/gpu:0"):
                dut = model(NUM_CLASSES)
                out = dut.forward(dummy_in)

            with tf.device("/cpu:0"):
                loss = cross_entropy(
                    tf.squeeze(dummy_gt, squeeze_dims=[3]), out, 1.)

            with tf.device("/gpu:0"):
                optimizer = tf.train.AdamOptimizer()
                grads = optimizer.compute_gradients(
                    loss, var_list=tf.trainable_variables())
                train_op = optimizer.apply_gradients(grads)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                before = sess.run(tf.trainable_variables())
                pred, outsizes = sess.run(
                    (out, dut.outsizes),
                    feed_dict={
                        dummy_in: np.ones((1, IMAGE_SIZE, IMAGE_SIZE, 3)),
                    })
                sess.run(
                    train_op,
                    feed_dict={
                        dummy_in: np.ones((1, IMAGE_SIZE, IMAGE_SIZE, 3)),
                        dummy_gt: np.ones((1, IMAGE_SIZE, IMAGE_SIZE, 1)),
                    })
                after = sess.run(tf.trainable_variables())
                for b, a in zip(before, after):
                    # Make sure something changed.
                    assert (b != a).any()

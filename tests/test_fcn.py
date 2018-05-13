"""Test set for FCN classes.
"""

import numpy as np
import tensorflow as tf

import project_root

from sss.models.fcn import FCN


def test_fcn_init():
    """Test FCN class initialization.
    """
    NUM_CLASSES = 5
    CKPT_PATH = 'dummy'

    with tf.Graph().as_default():
        # Check methods constructed in __init__().
        dut = FCN(NUM_CLASSES)
        assert dut.logger is not None
        assert dut.num_classes == NUM_CLASSES
        assert dut.vgg_pretrain_ckpt_path is None
        assert hasattr(dut, 'conv1_1')
        assert hasattr(dut, 'conv1_2')
        assert hasattr(dut, 'pool1')
        assert hasattr(dut, 'conv2_1')
        assert hasattr(dut, 'conv2_2')
        assert hasattr(dut, 'pool2')
        assert hasattr(dut, 'conv3_1')
        assert hasattr(dut, 'conv3_2')
        assert hasattr(dut, 'conv3_3')
        assert hasattr(dut, 'pool3')
        assert hasattr(dut, 'conv4_1')
        assert hasattr(dut, 'conv4_2')
        assert hasattr(dut, 'conv4_3')
        assert hasattr(dut, 'pool4')
        assert hasattr(dut, 'conv5_1')
        assert hasattr(dut, 'conv5_2')
        assert hasattr(dut, 'conv5_3')
        assert hasattr(dut, 'pool5')
        assert hasattr(dut, 'fc6')
        assert hasattr(dut, 'dropout6')
        assert hasattr(dut, 'fc7')
        assert hasattr(dut, 'dropout7')
        assert hasattr(dut, 'score_fr')
        assert hasattr(dut, 'upscore32')
        assert not hasattr(dut, 'upscore2')
        assert not hasattr(dut, 'score_pool4')
        assert not hasattr(dut, 'upscore4')
        assert not hasattr(dut, 'score_pool3')

        dut = FCN(NUM_CLASSES, mode='fcn16')
        assert hasattr(dut, 'upscore2')
        assert hasattr(dut, 'score_pool4')
        assert not hasattr(dut, 'upscore4')
        assert not hasattr(dut, 'score_pool3')

        dut = FCN(NUM_CLASSES, mode='fcn8')
        assert hasattr(dut, 'upscore2')
        assert hasattr(dut, 'score_pool4')
        assert hasattr(dut, 'upscore4')
        assert hasattr(dut, 'score_pool3')

        # checkpoint path specified.
        dut = FCN(NUM_CLASSES, vgg_pretrain_ckpt_path=CKPT_PATH)
        assert dut.vgg_pretrain_ckpt_path == CKPT_PATH


def test_fcn_architecture():
    """Test the architecture of FCN.
    """
    IMAGE_SIZE = 224
    MODE = 'fcn8'
    NUM_CLASSES = 5
    GT_VAR_LIST = sorted([
        "fcn/conv1/conv1_1/weights:0", "fcn/conv1/conv1_1/biases:0",
        "fcn/conv1/conv1_2/weights:0", "fcn/conv1/conv1_2/biases:0",
        "fcn/conv2/conv2_1/weights:0", "fcn/conv2/conv2_1/biases:0",
        "fcn/conv2/conv2_2/weights:0", "fcn/conv2/conv2_2/biases:0",
        "fcn/conv3/conv3_1/weights:0", "fcn/conv3/conv3_1/biases:0",
        "fcn/conv3/conv3_2/weights:0", "fcn/conv3/conv3_2/biases:0",
        "fcn/conv3/conv3_3/weights:0", "fcn/conv3/conv3_3/biases:0",
        "fcn/conv4/conv4_1/weights:0", "fcn/conv4/conv4_1/biases:0",
        "fcn/conv4/conv4_2/weights:0", "fcn/conv4/conv4_2/biases:0",
        "fcn/conv4/conv4_3/weights:0", "fcn/conv4/conv4_3/biases:0",
        "fcn/conv5/conv5_1/weights:0", "fcn/conv5/conv5_1/biases:0",
        "fcn/conv5/conv5_2/weights:0", "fcn/conv5/conv5_2/biases:0",
        "fcn/conv5/conv5_3/weights:0", "fcn/conv5/conv5_3/biases:0",
        "fcn/fc6/weights:0", "fcn/fc6/biases:0", "fcn/fc7/weights:0",
        "fcn/fc7/biases:0", "fcn/score_fr/weights:0", "fcn/score_fr/biases:0",
        "fcn/score_pool3/weights:0", "fcn/score_pool3/biases:0",
        "fcn/score_pool4/weights:0", "fcn/score_pool4/biases:0",
        "fcn/upscore2/weights:0", "fcn/upscore32/weights:0",
        "fcn/upscore4/weights:0"
    ])

    with tf.Graph().as_default():
        dut = FCN(NUM_CLASSES, mode=MODE)
        dummy_in = tf.placeholder(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE, 3))
        dut.forward(dummy_in)
        var_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES)
        var_list = sorted(var.name for var in var_list)
        assert var_list == GT_VAR_LIST


def test_fcn_update():
    """Test FCN surely updates the parameters.
    TODO: test not only FCN but all networks at once.
    cf. https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
    """
    IMAGE_SIZE = 224
    NUM_CLASSES = 5
    MODES = ('fcn32', 'fcn16', 'fcn8')

    for mode in MODES:
        with tf.device("/gpu:0"):
            with tf.Graph().as_default():
                dut = FCN(NUM_CLASSES, mode=mode)
                dummy_in = tf.placeholder(tf.float32,
                                          (None, IMAGE_SIZE, IMAGE_SIZE, 3))
                dummy_gt = tf.placeholder(
                    tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1])
                out = dut.forward(dummy_in)
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=out,
                        labels=tf.squeeze(dummy_gt, squeeze_dims=[3])))
                optimizer = tf.train.AdamOptimizer()
                grads = optimizer.compute_gradients(
                    loss, var_list=tf.trainable_variables())
                train_op = optimizer.apply_gradients(grads)
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    before = sess.run(tf.trainable_variables())
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

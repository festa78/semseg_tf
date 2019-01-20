"""Test set for model classes.
"""

import numpy as np
import tensorflow as tf

import project_root

from src.models.fcn import fcn32, fcn16, fcn8
from src.models.dilation_net import frontend, dilation7, dilation8, dilation10
from src.models.psp_net import pspnet50, pspnet101
from src.utils.losses import cross_entropy


class Test(tf.test.TestCase):

    def test_network_update(self):
        """Test networks surely updates the parameters.
        cf. https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
        """
        IMAGE_SIZE = 224
        NUM_CLASSES = 5
        LEARNING_RATE = .1
        MODELS = (fcn32, fcn16, fcn8, frontend, dilation7, dilation8,
                  dilation10, pspnet50, pspnet101)
        np.random.seed(1234)
        tf.set_random_seed(1234)

        for model in MODELS:
            with tf.device("/cpu:0"):
                dummy_in = tf.placeholder(tf.float32,
                                          (None, IMAGE_SIZE, IMAGE_SIZE, 3))
                dummy_gt = tf.placeholder(
                    tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1])

            with tf.device("/gpu:0"):
                dut = model(NUM_CLASSES)
                out = dut(dummy_in)

            with tf.device("/cpu:0"):
                loss = cross_entropy(dummy_gt, out, 1.)

            with tf.device("/gpu:0"):
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=LEARNING_RATE)
                grads = optimizer.compute_gradients(
                    loss, var_list=tf.trainable_variables())
                train_op = optimizer.apply_gradients(grads)

            with self.test_session() as sess:
                sess.run(tf.global_variables_initializer())
                before = sess.run(tf.trainable_variables(scope='model'))
                sess.run(
                    train_op,
                    feed_dict={
                        dummy_in:
                        np.random.rand(1, IMAGE_SIZE, IMAGE_SIZE, 3),
                        dummy_gt:
                        np.random.randint(
                            NUM_CLASSES, size=(1, IMAGE_SIZE, IMAGE_SIZE, 1)),
                    })
                after = sess.run(tf.trainable_variables(scope='model'))
                for i, (b, a) in enumerate(zip(before, after)):
                    # Make sure something changed.
                    self.assertTrue((b != a).any())

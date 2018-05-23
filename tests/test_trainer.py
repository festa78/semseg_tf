import numpy as np
import tensorflow as tf

from sss.models.fcn import FCN
from sss.pipelines.trainer import Trainer
from sss.utils.losses import cross_entropy


def test_trainer_update():
    """Test Trainer class surely updates the parameters.
    cf. https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
    """
    IMAGE_SIZE = 224
    NUM_CLASSES = 5

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            dummy_images = tf.data.Dataset.from_tensor_slices(
                tf.convert_to_tensor(
                    np.ones([4, IMAGE_SIZE, IMAGE_SIZE, 3]), dtype=tf.float32))
            dummy_labels = tf.data.Dataset.from_tensor_slices(
                tf.convert_to_tensor(
                    np.ones([4, IMAGE_SIZE, IMAGE_SIZE, 1]), dtype=tf.int64))
            dummy_files = tf.data.Dataset.from_tensor_slices(
                tf.convert_to_tensor(('1', '2', '3', '4'), dtype=tf.string))
            dataset = tf.data.Dataset.zip((dummy_images, dummy_labels, dummy_files))
            dataset = dataset.batch(2)
            iterator = dataset.make_one_shot_iterator()

        with tf.device('/gpu:0'):
            batch = iterator.get_next()
            model = FCN(NUM_CLASSES)

            optimizer = tf.train.AdamOptimizer()

            dut = Trainer(model, batch, cross_entropy, optimizer)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            before = sess.run(tf.trainable_variables())

            dut.train(sess)

            after = sess.run(tf.trainable_variables())

    for b, a in zip(before, after):
        # Make sure something changed.
        assert (b != a).any()

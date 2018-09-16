"""Test set for DataPreprocessor class.
"""

import numpy as np
import tensorflow as tf

from src.data.data_preprocessor import DataPreprocessor


def test_process_image():
    """Test DataPreprocessor.process_image.
    """
    dataset = tf.data.Dataset.from_tensor_slices({
        'image':
        tf.constant(np.zeros((3, 3, 3))),
        'label':
        tf.constant(np.zeros((3, 1)))
    })

    dut = DataPreprocessor(dataset)

    # Lambda function with the external parameter @p a.
    dut.process_image(lambda image, a: image + a + 1, a=1)

    batch = dut.dataset.make_one_shot_iterator().get_next()

    image_equal_op = tf.equal(batch['image'], tf.constant(np.ones((3, 3)) * 2))
    label_equal_op = tf.equal(batch['label'], tf.constant(np.zeros((1,))))

    with tf.Session() as sess:
        for _ in range(3):
            for op in sess.run((image_equal_op, label_equal_op)):
                assert np.all(op)


def test_process_label():
    """Test DataPreprocessor.process_label.
    """
    dataset = tf.data.Dataset.from_tensor_slices({
        'image':
        tf.constant(np.zeros((3, 3, 3))),
        'label':
        tf.constant(np.zeros((3, 1)))
    })

    dut = DataPreprocessor(dataset)

    # Lambda function with the external parameter @p a.
    dut.process_label(lambda label, a: label + a + 1, a=1)

    batch = dut.dataset.make_one_shot_iterator().get_next()

    image_equal_op = tf.equal(batch['image'], tf.constant(np.zeros((3, 3))))
    label_equal_op = tf.equal(batch['label'], tf.constant(np.ones((1,)) * 2))

    with tf.Session() as sess:
        for _ in range(3):
            for op in sess.run((image_equal_op, label_equal_op)):
                assert np.all(op)


def test_process_image_and_label():
    """Test DataPreprocessor.process_image_and_label.
    """
    dataset = tf.data.Dataset.from_tensor_slices({
        'image':
        tf.constant(np.zeros((3, 3, 3))),
        'label':
        tf.constant(np.zeros((3, 1)))
    })

    dut = DataPreprocessor(dataset)

    # Lambda function with the external parameter @p a.
    dut.process_image_and_label(
        lambda image, label, a: (image - a - 1, label + a + 1), a=1)

    batch = dut.dataset.make_one_shot_iterator().get_next()

    image_equal_op = tf.equal(batch['image'], tf.constant(-np.ones((3, 3)) * 2))
    label_equal_op = tf.equal(batch['label'], tf.constant(np.ones((1,)) * 2))

    with tf.Session() as sess:
        for _ in range(3):
            for op in sess.run((image_equal_op, label_equal_op)):
                assert np.all(op)

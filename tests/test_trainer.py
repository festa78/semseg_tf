import numpy as np
import tensorflow as tf

from sss.models.fcn import fcn32
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
            dummy_batch = {
                'image':
                tf.convert_to_tensor(
                    np.ones([4, IMAGE_SIZE, IMAGE_SIZE, 3]), dtype=tf.float32),
                'label':
                tf.convert_to_tensor(
                    np.ones([4, IMAGE_SIZE, IMAGE_SIZE, 1]), dtype=tf.int64),
                'filename':
                tf.convert_to_tensor(('1', '2', '3', '4'), dtype=tf.string)
            }
            dataset = tf.data.Dataset.from_tensor_slices(dummy_batch)
            dataset = dataset.batch(2)
            iterator = dataset.make_one_shot_iterator()
            batch = iterator.get_next()
            global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.device('/gpu:0'):
            model = fcn32(NUM_CLASSES)

            optimizer = tf.train.AdamOptimizer()

            dut = Trainer(model, batch, cross_entropy, optimizer, global_step)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            before = sess.run(tf.trainable_variables())

            dut.train(sess)

            after = sess.run(tf.trainable_variables())
            step = tf.train.global_step(sess, global_step)

    assert step == 2
    for b, a in zip(before, after):
        # Make sure something changed.
        assert (b != a).any()

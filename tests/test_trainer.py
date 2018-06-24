import numpy as np
import tensorflow as tf

from sss.models.fcn import fcn32
from sss.pipelines.trainer import Trainer
from sss.utils.losses import cross_entropy


def test_trainer_update(tmpdir):
    """Test Trainer class surely updates the parameters.
    cf. https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
    """
    BATCH_SIZE = 4
    IMAGE_SIZE = 224
    NUM_CLASSES = 5
    NUM_EPOCHS = 2
    EVALUATE_FREQ = 10

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            train_batch = {
                'image':
                tf.convert_to_tensor(
                    np.ones([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3]),
                    dtype=tf.float32),
                'label':
                tf.convert_to_tensor(
                    np.ones([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1]),
                    dtype=tf.int64),
                'height': tf.constant([IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE]),
                'width': tf.constant([IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE])
            }
            val_batch = {
                'image':
                tf.convert_to_tensor(
                    np.ones([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3]),
                    dtype=tf.float32),
                'label':
                tf.convert_to_tensor(
                    np.ones([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1]),
                    dtype=tf.int64),
                'height': tf.constant([IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE]),
                'width': tf.constant([IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE])
            }
            train_dataset = tf.data.Dataset.from_tensor_slices(train_batch)
            train_dataset = train_dataset.batch(2)
            train_iterator = train_dataset.make_initializable_iterator()
            val_dataset = tf.data.Dataset.from_tensor_slices(val_batch)
            val_dataset = val_dataset.batch(2)
            val_iterator = val_dataset.make_initializable_iterator()
            global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.device('/gpu:0'):
            model = fcn32(NUM_CLASSES)

            optimizer = tf.train.AdamOptimizer()

            dut = Trainer(
                model,
                NUM_CLASSES,
                train_iterator,
                val_iterator,
                cross_entropy,
                optimizer,
                global_step,
                str(tmpdir),
                num_epochs=NUM_EPOCHS,
                evaluate_freq=EVALUATE_FREQ)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            before = sess.run(tf.trainable_variables(scope='model'))

            dut.train(sess)

            after = sess.run(tf.trainable_variables(scope='model'))

    for b, a in zip(before, after):
        # Make sure something changed.
        assert (b != a).any()

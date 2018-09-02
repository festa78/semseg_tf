import numpy as np
import tensorflow as tf

from sss.data.cityscapes import trainid2color_tensor
from sss.models.fcn import fcn32
from sss.pipelines.trainer import Trainer
from sss.utils.losses import cross_entropy


def _setup_trainer(tmpdir):
    """Setup a simple Trainer class instance.
    """
    BATCH_SIZE = 4
    IMAGE_SIZE = 224
    NUM_CLASSES = 5
    NUM_EPOCHS = 2
    CLASS_WEIGHTS = tf.constant(
        np.array([.1, .3, .2, .1, .3], dtype=np.float), dtype=tf.float32)
    EVALUATE_FREQ = 10

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
            'height':
            tf.constant([IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE]),
            'width':
            tf.constant([IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE])
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
            'height':
            tf.constant([IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE]),
            'width':
            tf.constant([IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE])
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
            trainid2color_tensor,
            str(tmpdir),
            train_class_weights=CLASS_WEIGHTS,
            num_epochs=NUM_EPOCHS,
            evaluate_freq=EVALUATE_FREQ)

    return dut


def test_trainer_update(tmpdir):
    """Test Trainer class surely updates the parameters.
    cf. https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
    """
    with tf.Graph().as_default():
        tf.set_random_seed(1234)
        dut = _setup_trainer(tmpdir)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            before = sess.run(tf.trainable_variables(scope='model'))

            dut.train(sess)

            after = sess.run(tf.trainable_variables(scope='model'))

    for b, a in zip(before, after):
        # Make sure something changed.
        assert (b != a).any()


def test_compute_metrics(tmpdir):
    """Test Trainer class surely compute mean loss and IoU.
    """
    with tf.Graph().as_default():
        tf.set_random_seed(1234)
        dut = _setup_trainer(tmpdir)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run((dut.train_iterator.initializer,
                      dut.train_metric_reset_op))

            train_mloss, train_miou, _ = sess.run(
                (dut.train_mean_loss, dut.train_mean_iou,
                 dut.train_epoch_summary_op))

            # Without update, it should be zero.
            assert train_mloss == 0.
            assert train_miou == 0.

            step_op = (dut.train_mean_loss_update_op,
                       dut.train_mean_iou_update_op, dut.train_op)
            out = sess.run(step_op)

            train_mloss, train_miou = sess.run((
                dut.train_mean_loss,
                dut.train_mean_iou,
            ))

            # After update.
            np.testing.assert_almost_equal(train_mloss, 1027812.875)
            np.testing.assert_almost_equal(train_miou, 0.02145448)


def test_compute_class_weights(tmpdir):
    """Test Trainer class surely compute class weights.
    """
    with tf.Graph().as_default():
        tf.set_random_seed(1234)
        dut = _setup_trainer(tmpdir)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(dut.train_iterator.initializer)
            weights = sess.run(dut.train_class_weights)
            assert np.allclose(weights, np.ones(weights.shape) * .3)

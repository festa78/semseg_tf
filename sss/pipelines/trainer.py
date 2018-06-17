"""Training pipeline for semantic segmentation.
"""

import logging
logging.basicConfig(level=logging.INFO)
import time

import tensorflow as tf

from sss.data.cityscapes import trainid2color_tensor


class Trainer:
    """Basic training pipeline class which integrate
    model, data, loss, and optimizer and
    start semantic segmentation training.
    This class supposed to be instantiated on gpu.

    Parameters
    ----------
    model: object
        A semantic segmentation model object which
        has .forward(input) method to get model output.
    batch: tf.Tensor
        A nested structure of tf.Tensor objects where
        the each object contain batch data to train.
        Supposed to be constructed by tf.Iterator.get_next().
    loss_fn: functional
        A functional which outputs loss value according to
        the same sized inputs tensors: predicted output
        tf.Tensor, ground truth output tf.Tensor,
        and weight tf.Tensor which weights losses
        on each pixel when conducting reduce mean operation.
    class_weights: tf.Tensor
        Weights to losses over classes.
        This array will be used as the parameter of @p loss_fn.
    optimizer: tf.Train.Optimizer
        A optimizer class which optimizes parameters of
        the @p model with losses computed by @loss_fn.
    global_step: tf.Variable
        A global step value to use with optimizer and
        logging purpose.
    logdir: str
        A path to the directory to save a summary to
        visualize on TensorBoard.
    """

    def __init__(self, model, batch, loss_fn, class_weights, optimizer,
                 global_step, logdir):
        # Inspect inputs.
        if hasattr(model, 'forward') is False:
            raise AttributeError('model object should have .forward() method.')

        self.logger = logging.getLogger(__name__)
        self.model = model
        self.batch = batch
        self.loss_fn = loss_fn
        self.class_weights = class_weights
        self.optimizer = optimizer
        self.global_step = global_step
        self.logdir = logdir
        self.logits = model.forward(self.batch['image'])
        self.predictions = tf.argmax(self.logits, axis=3)

        # Loss computation should always live in cpu.
        # TODO: Can it be in gpu?
        with tf.device('cpu:0'):
            # Construct class weights tensor.
            self.class_weights_tensor = tf.cast(
                tf.reshape(self.batch['label'], shape=[-1]), dtype=tf.float32)
            self.class_weights_tensor = tf.map_fn(
                lambda x: self.class_weights[tf.cast(x, dtype=tf.int64)],
                self.class_weights_tensor)
            self.class_weights_tensor = tf.reshape(
                self.class_weights_tensor, shape=tf.shape(self.batch['label']))
            self.class_weights_tensor = tf.cast(
                self.class_weights_tensor, dtype=tf.float32)
            # Ignore label id: 0.
            ignore_mask = tf.cast(
                tf.not_equal(self.batch['label'], 0), tf.float32)
            self.class_weights_tensor = tf.multiply(self.class_weights_tensor,
                                                    ignore_mask)

            self.loss = self.loss_fn(
                tf.squeeze(self.batch['label'], squeeze_dims=[3]), self.logits,
                self.class_weights_tensor)

            # Add to summary.
            image_summary = tf.concat(
                (self.batch['image'][0] * 255.,
                 trainid2color_tensor(self.batch['label'][0]),
                 trainid2color_tensor(self.predictions[0])),
                axis=1)
            tf.summary.image('visualization', tf.expand_dims(image_summary, 0))
            tf.summary.scalar('train_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

        self.train_op = self.optimizer.minimize(
            self.loss,
            var_list=tf.trainable_variables(),
            global_step=self.global_step)

    def train(self, sess):
        train_writer = tf.summary.FileWriter(self.logdir, sess.graph)
        while True:
            try:
                start = time.clock()
                summary, loss, _ = sess.run(
                    [self.summary_op, self.loss, self.train_op])
                proc_time = time.clock() - start
                step = tf.train.global_step(sess, self.global_step)
                self.logger.info(
                    'step: {},\tproc_time: {:06f},\tloss: {:06f}'.format(
                        step, proc_time, loss))
                train_writer.add_summary(summary, step)
            except tf.errors.OutOfRangeError:
                break

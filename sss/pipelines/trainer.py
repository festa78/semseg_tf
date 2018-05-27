"""Training pipeline for semantic segmentation.
"""

import tensorflow as tf


class Trainer:
    """Training pipeline class.
    """

    def __init__(self, model, batch, loss_fn, optimizer, global_step):
        self.model = model
        self.images, self.labels, self.filenames = batch
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.global_step = global_step
        self.predictions = model.forward(self.images)
        # Loss computation should always live in cpu.
        # TODO: Can it be in gpu?
        with tf.device('cpu:0'):
            self.loss = self.loss_fn(self.predictions, self.labels)
        self.train_op = self.optimizer.minimize(
            self.loss,
            var_list=tf.trainable_variables(),
            global_step=self.global_step)

    def train(self, sess):
        while True:
            try:
                loss, _ = sess.run([self.loss, self.train_op])
                print('step:', tf.train.global_step(sess, self.global_step),
                      'loss:', loss)
            except tf.errors.OutOfRangeError:
                break

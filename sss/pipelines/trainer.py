"""Training pipeline for semantic segmentation.
"""

import tensorflow as tf


class Trainer:
    """Training pipeline class.
    """

    def __init__(self, model, batch, loss_fn, optimizer):
        self.model = model
        self.images, self.labels, self.filenames = batch
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.predictions = model.forward(self.images)
        # Loss computation should always live in cpu.
        # TODO: Can it be in gpu?
        with tf.device('cpu:0'):
            self.loss = self.loss_fn(self.predictions, self.labels)
        self.train_op = self.optimizer.minimize(self.loss, var_list=tf.trainable_variables())

    def train(self, sess):
        while True:
            try:
                sess.run(self.train_op)
                print('train')
            except tf.errors.OutOfRangeError:
                break

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
    num_classes: int
        The number of output classes of the model.
    train_iterator: tf.Tensor
        The initializable iterator for training.
        .get_next() is used to create train batch operator.
    val_iterator: tf.Tensor
        The initializable iterator for validation.
        .get_next() is used to create validation batch operator.
    loss_fn: functional
        A functional which outputs loss value according to
        the same sized inputs tensors: predicted output
        tf.Tensor, ground truth output tf.Tensor,
        and weight tf.Tensor which weights losses
        on each pixel when conducting reduce mean operation.
    optimizer: tf.Train.Optimizer
        A optimizer class which optimizes parameters of
        the @p model with losses computed by @loss_fn.
    global_step: tf.Variable
        A global step value to use with optimizer and
        logging purpose.
    logdir: str
        A path to the directory to save a summary to
        visualize on TensorBoard.
    train_class_weights: 1d tf.Tensor, default None
        Weights to train losses over classes.
        This array will be used as the parameter of @p loss_fn.
        It should have 1d tensor with the length of the number of classes.
        If it's None, use 1 to equally weight classes.
    val_class_weights: 1d tf.Tensor, default None
        Weights to validation losses over classes.
        This array will be used as the parameter of @p loss_fn.
        It should have 1d tensor with the length of the number of classes.
        If it's None, use 1 to equally weight classes.
    num_epochs: int, default: 200
        The number epochs to train.
    evaluate_freq: int, default: 10
        Evaluate model by validation dataset and compute metrics
        every @p evaluate_freq epochs.
    """

    def __init__(self,
                 model,
                 num_classes,
                 train_iterator,
                 val_iterator,
                 loss_fn,
                 optimizer,
                 global_step,
                 logdir,
                 train_class_weights=None,
                 val_class_weights=None,
                 num_epochs=200,
                 evaluate_freq=10):
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.num_classes = num_classes
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.train_batch = self.train_iterator.get_next()
        self.val_batch = self.val_iterator.get_next()
        self.train_image_height = self.train_batch['height']
        self.train_image_width = self.train_batch['width']
        self.val_image_height = self.val_batch['height']
        self.val_image_width = self.val_batch['width']
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.global_step = global_step
        self.logdir = logdir
        self.num_epochs = num_epochs
        self.evaluate_freq = evaluate_freq

        # Inspect inputs.
        if hasattr(model, 'forward') is False:
            raise AttributeError('model object should have .forward() method.')
        if any(key not in self.train_batch for key in ('image', 'label')):
            raise AttributeError(
                'train_batch object should have "image" and "label" keys')
        if any(key not in self.val_batch for key in ('image', 'label')):
            raise AttributeError(
                'val_batch object should have "image" and "label" keys')

        self.train_class_weights, \
            self.train_loss, \
            self.train_image_summary, \
            self.train_mean_loss, \
            self.train_mean_loss_update_op, \
            self.train_mean_iou, \
            self.train_mean_iou_update_op, \
            self.train_metric_reset_op, \
            self.train_step_summary_op, \
            self.train_epoch_summary_op = self.compute_metrics(
                'train', self.train_batch['image'], self.train_batch['label'], train_class_weights
                )

        self.val_class_weights, \
            self.val_loss, \
            self.val_image_summary, \
            self.val_mean_loss, \
            self.val_mean_loss_update_op, \
            self.val_mean_iou, \
            self.val_mean_iou_update_op, \
            self.val_metric_reset_op, \
            self.val_step_summary_op, \
            self.val_epoch_summary_op = self.compute_metrics(
                'val', self.val_batch['image'], self.val_batch['label'], val_class_weights
                )

        self.train_op = self.optimizer.minimize(
            self.train_loss,
            var_list=tf.trainable_variables(scope='model'),
            global_step=self.global_step)

    def compute_metrics(self, mode, image, label, class_weights):
        with tf.variable_scope('model'):
            logits = self.model.forward(image)
            predictions = tf.argmax(logits, axis=3)

        # Metric computations should live in cpu.
        with tf.device('cpu:0'):
            class_weights_tensor = self.compute_class_weights(
                label, class_weights=class_weights)

            with tf.variable_scope('{}_step_metrics'.format(mode)) as scope:
                loss = self.loss_fn(
                    tf.squeeze(label, squeeze_dims=[3]), logits,
                    class_weights_tensor)

                image_summary = tf.concat(
                    (image[0] * 255., trainid2color_tensor(label[0]),
                     trainid2color_tensor(predictions[0])),
                    axis=1)

            with tf.variable_scope('{}_epoch_metrics'.format(mode)) as scope:
                mean_loss, mean_loss_update_op = tf.metrics.mean(loss)
                mean_iou, mean_iou_update_op = tf.metrics.mean_iou(
                    tf.squeeze(label, squeeze_dims=[3]),
                    predictions,
                    num_classes=self.num_classes,
                    weights=class_weights_tensor)
                var_list = tf.contrib.framework.get_variables(
                    scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
                metric_reset_op = tf.variables_initializer(var_list)

            # Add to summary.
            # NOTE: Needs to separate epoch metric summaries as they need to compute
            # after the update operations.
            step_summaries = []
            epoch_summaries = []
            step_summaries.append(
                tf.summary.image('{}_visualization'.format(mode),
                                 tf.expand_dims(image_summary, 0)))
            step_summaries.append(
                tf.summary.scalar('{}_mean_loss'.format(mode), mean_loss))
            epoch_summaries.append(
                tf.summary.scalar('{}_mean_iou'.format(mode), mean_iou))

            step_summary_op = tf.summary.merge(step_summaries)
            epoch_summary_op = tf.summary.merge(epoch_summaries)

        return class_weights_tensor, loss, image_summary, mean_loss, mean_loss_update_op, \
            mean_iou, mean_iou_update_op, metric_reset_op, \
            step_summary_op, epoch_summary_op

    def compute_class_weights(self, label, class_weights=None, ignore_id=0):
        """Compute weights on loss by label data.

        Parameters
        ----------
        label: (N, H, W, C=1) tf.tensor
            Label batch used as a ground truth.
        class_weights: 1d tf.Tensor, default None
            Weights to losses over classes.
            This array will be used as the parameter of @p self.loss_fn.
            It should have 1d tensor with the length of the number of classes.
            If it's None, use 1 to equally weight classes.
        ignore_id: int, default 0
            Set weight as 0 if class id is @p ignore_id.
            This means it will ignore losses of @p ignore_id class.

        Returns
        -------
        class_weights_tensor: (N, H, W, C=1) tf.tensor
            Constructed weights tensor.
            This has the same shape to @p label.
        """
        if class_weights is not None:
            self.class_weights_tensor = tf.gather(class_weights, label)
        else:
            self.class_weights_tensor = tf.ones_like(label, dtype=tf.float32)

        ignore_mask = tf.cast(tf.not_equal(label, ignore_id), tf.float32)
        class_weights_tensor = tf.multiply(self.class_weights_tensor,
                                           ignore_mask)
        return class_weights_tensor

    def train(self, sess):
        """Execute train loop.

        Parameters
        ----------
        sess: tf.Session
            TensorFlow session to run train loop.
        """
        self.logger.info('Start training.')
        summary_writer = tf.summary.FileWriter(self.logdir, sess.graph)
        for ep in range(self.num_epochs):
            sess.run((self.train_iterator.initializer,
                      self.train_metric_reset_op))
            if (ep + 1) % self.evaluate_freq == 0:
                step_op = (self.train_mean_loss_update_op,
                           self.train_mean_iou_update_op,
                           self.train_step_summary_op, self.train_op)
            else:
                step_op = self.train_op

            start = time.clock()
            while True:
                try:
                    out = sess.run(step_op)
                except tf.errors.OutOfRangeError:
                    break
            proc_time = time.clock() - start

            self.logger.info('Train epoch: {},\tproc time: {:06f}'.format(
                ep, proc_time))

            if (ep + 1) % self.evaluate_freq == 0:
                _, _, train_step_summary, _ = out
                train_mloss, train_miou, train_epoch_summary = sess.run(
                    (self.train_mean_loss, self.train_mean_iou,
                     self.train_epoch_summary_op))
                self.logger.info(
                    'Train mean loss: {:06f},\ttrain mean IoU: {:06f}'.format(
                        train_mloss, train_miou))
                summary_writer.add_summary(train_step_summary, ep)
                summary_writer.add_summary(train_epoch_summary, ep)
                self.validate(sess, summary_writer, ep)

    def validate(self, sess, summary_writer, epoch):
        """Execute validation loop.

        Parameters
        ----------
        summary_writer: tf.summary.FileWriter
            The summary writer to add summary metrics of validation.
        """
        self.logger.info('Start evaluation.')
        sess.run((self.val_iterator.initializer, self.val_metric_reset_op))
        step_op = (self.val_mean_loss_update_op, self.val_mean_iou_update_op,
                   self.val_step_summary_op)

        start = time.clock()
        while True:
            try:
                out = sess.run(step_op)
            except tf.errors.OutOfRangeError:
                break
        proc_time = time.clock() - start

        self.logger.info('Validation proc time: {:06f}'.format(proc_time))
        _, _, val_step_summary = out
        val_mloss, val_miou, val_epoch_summary = sess.run(
            (self.val_mean_loss, self.val_mean_iou, self.val_epoch_summary_op))
        self.logger.info(
            'Validation mean loss: {:06f},\tvalidation mean IoU: {:06f}'.format(
                val_mloss, val_miou))
        summary_writer.add_summary(val_step_summary, epoch)
        summary_writer.add_summary(val_epoch_summary, epoch)

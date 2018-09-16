"""Training pipeline for semantic segmentation.
"""

import logging
logging.basicConfig(level=logging.INFO)
import os
import time

import tensorflow as tf


class Trainer:
    """Basic training pipeline class which integrate
    model, data, loss, and optimizer and
    start semantic segmentation training.
    This class supposed to be instantiated on gpu.

    Parameters
    ----------
    model: object
        A semantic segmentation model object which
        has .__call__(input) method to get model output.
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
    color_map_fn: functional
        A functional which can convert a class id to
        a corresponding rgb color.
        E.g. src.data.cityscapes.trainid2color_tensor.
    save_dir: str
        A path to the directory to save logs and models.
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
    evaluate_epochs: int, default: 10
        Evaluate model by validation dataset and save the session
        every @p evaluate_epochs epochs.
    verbose_steps: int, default: 10
        Show metric every @p verbose_step.
    resume_path: str, default: None
        The path to resume session from.
    finetune_from: str, default: None
        If specified, resume only model weights from the architecture.
    """

    def __init__(self,
                 model,
                 num_classes,
                 train_iterator,
                 val_iterator,
                 loss_fn,
                 optimizer,
                 global_step,
                 color_map_fn,
                 save_dir,
                 train_class_weights=None,
                 val_class_weights=None,
                 num_epochs=200,
                 evaluate_epochs=10,
                 verbose_steps=10,
                 resume_path=None,
                 finetune_from=None):
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
        self.color_map_fn = color_map_fn
        self.save_dir = save_dir
        self.num_epochs = num_epochs
        self.evaluate_epochs = evaluate_epochs
        self.verbose_steps = verbose_steps
        self.resume_path = resume_path
        self.finetune_from = finetune_from

        # Inspect inputs.
        if hasattr(model, '__call__') is False:
            raise AttributeError('model object should have .__call__() method.')
        if hasattr(optimizer, 'minimize') is False:
            raise AttributeError(
                'optimizer object should have .minimize() method.')
        if any(key not in self.train_batch for key in ('image', 'label')):
            raise AttributeError(
                'train_batch object should have "image" and "label" keys')
        if any(key not in self.val_batch for key in ('image', 'label')):
            raise AttributeError(
                'val_batch object should have "image" and "label" keys')

        # Set up metrics.
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
                self.train_batch['image'], self.train_batch['label'], 'train', train_class_weights
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
                self.val_batch['image'], self.val_batch['label'], 'val', val_class_weights
                )

        self.train_op = self.optimizer.minimize(
            self.train_loss,
            var_list=tf.trainable_variables(scope='model'),
            global_step=self.global_step)

        # Epoch ops and a saver should live in cpu.
        with tf.device('/cpu'):
            # In the training loop, it will increment epoch first.
            # So set -1 as the initial value to start from 0.
            self.epoch = tf.Variable(-1, name='epoch', trainable=False)
            self.increment_epoch_op = tf.assign(self.epoch, self.epoch + 1)
            self.epoch_less_than_max = tf.less(self.epoch,
                                               tf.constant(self.num_epochs))
            self.saver = tf.train.Saver()

        self.log_dir = os.path.join(self.save_dir, 'logs')
        self.ckpt_dir = os.path.join(self.save_dir, 'ckpts')
        os.makedirs(self.log_dir)
        os.makedirs(self.ckpt_dir)

    def compute_metrics(self,
                        image,
                        label,
                        name,
                        class_weights=None,
                        ignore_id=255):
        """Compute necessary metics: loss, weights, IoU, and summaries.

        Parameters
        ----------
        image: (N, H, W, C=3) tf.tensor
            Image batch used as an input.
        label: (N, H, W, C=1) tf.tensor
            Label batch used as a ground truth.
        name: string
            Variable scope prefix for the metrics.
        class_weights: 1d tf.Tensor, default None
            Weights to validation losses over classes.
            This array will be used as the parameter of @p loss_fn.
            It should have 1d tensor with the length of the number of classes.
            If it's None, use 1 to equally weight classes.
        ignore_id: int, default 255
            Set weight as 0 if class id is @p ignore_id.
            This means it will ignore losses of @p ignore_id class.

        Returns
        -------
        class_weights_tensor: (N, H, W, C=1) tf.tensor
            Constructed weights tensor.
            This has the same shape to @p label.
        loss: Scalar tensor
            Loss value.
        image_summary: (H, 3 * W, C=3) tf.tensor
            Set of {input, prediction, ground truth} image
            used for visualization.
        mean_loss: Scalar tensor
            Mean loss value per epoch.
        mean_loss_update_op:
            Operator to update mean loss.
        mean_iou: Scalar tensor
            Mean IoU per epoch.
        mean_iou_update_op:
            Operator to update mean IoU.
        metric_reset_op:
            Operator to reset mean_loss and mean_iou values.
            Supposed to be called every epoch.
        step_summary_op:
            Operator to compute metrics for each step.
        epoch_summary_op:
            Operator to compute metrics for each epoch.
        """
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            logits = self.model(image)
            predictions = tf.argmax(logits, axis=3)

        # Metric computations should live in cpu.
        with tf.device('cpu:0'):
            ignore_mask = tf.not_equal(label, ignore_id)
            class_weights_tensor = self.compute_class_weights(
                label, ignore_mask, class_weights)

            with tf.variable_scope('{}_step_metrics'.format(name)) as scope:
                loss = self.loss_fn(label, logits, class_weights_tensor)

                image_summary = tf.concat(
                    (image[0] * 255., self.color_map_fn(label[0]),
                     self.color_map_fn(tf.expand_dims(predictions[0], -1))),
                    axis=1)

            with tf.variable_scope('{}_epoch_metrics'.format(name)) as scope:
                label_ignored = tf.multiply(label, tf.cast(
                    ignore_mask, tf.int64))
                mean_loss, mean_loss_update_op = tf.metrics.mean(loss)
                mean_iou, mean_iou_update_op = tf.metrics.mean_iou(
                    tf.squeeze(label_ignored, axis=[3]),
                    predictions,
                    num_classes=self.num_classes,
                    weights=tf.cast(ignore_mask, tf.float32))
                var_list = tf.contrib.framework.get_variables(
                    scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
                metric_reset_op = tf.variables_initializer(var_list)

            # Add to summary.
            # NOTE: Needs to separate epoch metric summaries as they need to compute
            # after the update operations.
            step_summaries = []
            epoch_summaries = []
            step_summaries.append(
                tf.summary.image('{}_visualization'.format(name),
                                 tf.expand_dims(image_summary, 0)))
            step_summaries.append(
                tf.summary.scalar('{}_mean_loss'.format(name), mean_loss))
            epoch_summaries.append(
                tf.summary.scalar('{}_mean_iou'.format(name), mean_iou))

            step_summary_op = tf.summary.merge(step_summaries)
            epoch_summary_op = tf.summary.merge(epoch_summaries)

        return class_weights_tensor, loss, image_summary, mean_loss, mean_loss_update_op, \
            mean_iou, mean_iou_update_op, metric_reset_op, \
            step_summary_op, epoch_summary_op

    def compute_class_weights(self, label, ignore_mask, class_weights=None):
        """Compute weights on loss by label data.

        Parameters
        ----------
        label: (N, H, W, C=1) tf.tensor
            Label batch used as a ground truth.
        ignore_mask: (N, H, W, C=1) tf.tensor
            A boolean mask to ignore subset of elements.
            False elements will be ignored and set as 0 in
            @p class_weights_tensor.
        class_weights: 1d tf.Tensor, default None
            Weights to validation losses over classes.
            This array will be used as the parameter of @p loss_fn.
            It should have 1d tensor with the length of the number of classes.
            If it's None, use 1 to equally weight classes.

        Returns
        -------
        class_weights_tensor: (N, H, W, C=1) tf.tensor
            Constructed weights tensor.
            This has the same shape to @p label.
        """
        if class_weights is None:
            class_weights_tensor = tf.cast(ignore_mask, tf.float32)
            return class_weights_tensor

        # Temporary set ignore id to 0 to avoid a tf.gather error on id 255.
        label_tmp = tf.multiply(label, tf.cast(ignore_mask, tf.int64))
        class_weights_tensor = tf.gather(class_weights, label_tmp)
        # Set weight 0 for ignore id.
        class_weights_tensor = tf.multiply(class_weights_tensor,
                                           tf.cast(ignore_mask, tf.float32))

        return class_weights_tensor

    def train(self, sess):
        """Execute train loop.

        Parameters
        ----------
        sess: tf.Session
            TensorFlow session to run train loop.
        """
        if self.resume_path:
            if self.finetune_from:
                self.model._restore_model_variables(sess, self.resume_path, self.finetune_from)
                self.logger.info('Finetune from {}.'.format(self.finetune_from))
            else:
                self.saver.restore(sess, self.resume_path)
                self.logger.info('The session restored from {}.'.format(
                    self.resume_path))

        self.logger.info('Start training.')
        summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
        while sess.run(self.epoch_less_than_max):
            sess.run(self.increment_epoch_op)
            ep = sess.run(self.epoch)

            sess.run((self.train_iterator.initializer,
                      self.train_metric_reset_op))

            step_op = (self.global_step, self.train_step_summary_op,
                       self.train_mean_loss_update_op,
                       self.train_mean_iou_update_op, self.train_op)

            start = time.clock()
            while True:
                try:
                    out = sess.run(step_op)
                    step = out[0]
                    if step % self.verbose_steps == 0:
                        train_mloss, train_miou = sess.run(
                            (self.train_mean_loss, self.train_mean_iou))
                        self.logger.info(
                            'Train step: {}, mean loss: {:06f},\tmean iou: {:06f}'.
                            format(step, train_mloss, train_miou))
                except tf.errors.OutOfRangeError:
                    break
            proc_time = time.clock() - start

            # Avoid sess.run(self.train_step_summary_op) here, otherwise get OutOfRangeError.
            train_step_summary = out[1]
            train_mloss, train_miou, train_epoch_summary = sess.run(
                (self.train_mean_loss, self.train_mean_iou,
                 self.train_epoch_summary_op))
            self.logger.info(
                'Train epoch: {},\tproc time: {:06f}\tmean loss: {:06f},\ttrain mean IoU: {:06f}'.
                format(ep, proc_time, train_mloss, train_miou))
            summary_writer.add_summary(train_step_summary, ep)
            summary_writer.add_summary(train_epoch_summary, ep)

            if (ep + 1) % self.evaluate_epochs == 0:
                with tf.device('/cpu'):
                    save_path = '{:08d}.ckpt'.format(ep)
                    self.saver.save(sess, os.path.join(self.ckpt_dir,
                                                       save_path))
                    self.logger.info('The session saved')

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

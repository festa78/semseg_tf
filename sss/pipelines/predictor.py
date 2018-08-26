"""Prediction pipeline for semantic segmentation.
"""

import logging
logging.basicConfig(level=logging.INFO)
import os
import time

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


class Predictor:
    """Basic prediction pipeline class which integrate
    model, data, loss, and optimizer and
    start semantic segmentation prediction.
    This class supposed to be instantiated on gpu.

    Parameters
    ----------
    model: object
        A semantic segmentation model object which
        has .__call__(input) method to get model output.
    num_classes: int
        The number of output classes of the model.
    test_iterator: tf.Tensor
        The initializable iterator for testing.
        .get_next() is used to create test batch operator.
    ckpt_path: str, default: None
        The path to resume model ckpt from.
    color_map_fn: functional
        A functional which can convert a class id to
        a corresponding rgb color.
        E.g. sss.data.cityscapes.trainid2color_tensor.
    save_dir: str
        A path to the directory to save logs and models.
    """

    def __init__(
            self,
            model,
            num_classes,
            test_iterator,
            ckpt_path,
            color_map_fn,
            save_dir,
    ):
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.num_classes = num_classes
        self.test_iterator = test_iterator
        self.test_batch = self.test_iterator.get_next()
        self.test_filenames = self.test_batch['filename']
        self.ckpt_path = ckpt_path
        self.color_map_fn = color_map_fn
        self.save_dir = save_dir

        # Inspect inputs.
        if hasattr(model, '__call__') is False:
            raise AttributeError('model object should have .__call__() method.')
        if any(key not in self.test_batch for key in ('image', 'label')):
            raise AttributeError(
                'test_batch object should have "image" and "label" keys')

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            # Necessary for some special layers, e.g. batch normalization.
            if hasattr(model, 'training'):
                self.model.training = False
            self.logits = self.model(self.test_batch['image'])
            self.predictions = tf.argmax(self.logits, axis=3)

        with tf.device('cpu:0'):
            self.colors = self.color_map_fn(
                tf.expand_dims(self.predictions, -1))
            self.saver = tf.train.Saver(
                tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='model'))

        self.id_dir = os.path.join(self.save_dir, 'id')
        self.color_dir = os.path.join(self.save_dir, 'color')
        os.makedirs(self.id_dir)
        os.makedirs(self.color_dir)

    def prediction(self, sess):
        """Execute prediction loop.

        Parameters
        ----------
        sess: tf.Session
            TensorFlow session to run prediction loop.
        """

        self.saver.restore(sess, self.ckpt_path)
        self.logger.info('The session restored from {}.'.format(self.ckpt_path))

        self.logger.info('Start prediction.')

        sess.run(self.test_iterator.initializer)

        count = 0
        while True:
            try:
                start = time.clock()
                predictions, colors, filenames = sess.run(
                    (self.predictions, self.colors, self.test_filenames))
                proc_time = time.clock() - start
                self.logger.info('batch size: {}, proc time: {:06f}'.format(
                    len(predictions), proc_time))
                for prediction, color, filename in zip(predictions, colors,
                                                       filenames):
                    label_id = Image.fromarray(prediction.astype(np.uint8))
                    label_color = Image.fromarray(color.astype(np.uint8))
                    file_basename = os.path.splitext(
                        os.path.basename(str(filename)))[0]
                    label_id.save(
                        os.path.join(self.id_dir,
                                     '{}.png'.format(file_basename)))
                    label_color.save(
                        os.path.join(self.color_dir,
                                     '{}.png'.format(file_basename)))
                    count += 1

            except tf.errors.OutOfRangeError:
                break

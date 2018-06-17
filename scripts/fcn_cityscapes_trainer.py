#!/usr/bin/python3 -B
"""The script to train cityscapes data by a fully convolutional network (aka FCN).
"""

import argparse
import logging
logging.basicConfig(level=logging.INFO)
import os

import numpy as np
import tensorflow as tf
import yaml

import project_root

from sss.data.cityscapes import id2trainid_tensor
from sss.data.data_preprocessor import DataPreprocessor
from sss.data.tfrecord import read_tfrecord
from sss.models.fcn import fcn8, fcn16, fcn32
from sss.pipelines.trainer import Trainer
from sss.utils.image_processing import random_crop_image_and_label, random_flip_left_right_image_and_label
from sss.utils.losses import cross_entropy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "The script to train cityscapes data by a fully convolutional network (aka FCN)."
    )
    parser.add_argument(
        'params_yaml',
        type=str,
        help='Path to the parameter setting yaml file.')

    args = parser.parse_args()
    options = yaml.load(open(args.params_yaml))

    # Initializations.
    tf.set_random_seed(1234)
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    logdir_fullpath = os.path.expanduser(options['logdir'])
    if not os.path.isdir(logdir_fullpath):
        raise AttributeError('--logdir should be existing directory path.')

    # Data part should live in cpu.
    with tf.device('/cpu:0'):
        # Read from tfrecord format data made by sss.data.tfrecord.TFRecordWriter.
        train_dataset = read_tfrecord(
            os.path.join(options['tfdata_dir'], 'train.tfrecord'))
        train_data_processor = DataPreprocessor(
            dataset=train_dataset,
            num_parallel_calls=options['num_parallel_calls'],
            batch_size=options['batch_size'],
            max_epochs=None,
            shuffle_buffer_size=options['shuffle_buffer_size'],
            prefetch_buffer_size=options['prefetch_buffer_size'])

        val_dataset = read_tfrecord(
            os.path.join(options['tfdata_dir'], 'val.tfrecord'))
        val_data_processor = DataPreprocessor(
            dataset=val_dataset,
            num_parallel_calls=10,
            batch_size=1,
            max_epochs=1,
            shuffle_buffer_size=None,
            prefetch_buffer_size=None)

        # Add more pre-procesing blocks.
        if options['random_saturation'] > 0. and options['random_saturation'] < 1.:
            logging.info('Randomly adjust saturation by factor {}'.format(
                options['random_saturation']))
            lower = 1. - options['random_saturation']
            upper = 1. + options['random_saturation']
            train_data_processor.process_image(
                tf.image.random_saturation, lower=lower, upper=upper)
        if options['random_contrast'] > 0. and options['random_contrast'] < 1.:
            logging.info('Randomly adjust contrast by factor {}'.format(
                options['random_contrast']))
            lower = 1. - options['random_contrast']
            upper = 1. + options['random_contrast']
            train_data_processor.process_image(
                tf.image.random_contrast, lower=lower, upper=upper)
        if options['random_hue'] > 0. and options['random_hue'] < .5:
            logging.info('Randomly adjust hue by factor {}'.format(
                options['random_hue']))
            train_data_processor.process_image(
                tf.image.random_hue, max_delta=options['random_hue'])
        if options['random_brightness'] > 0. and options['random_brightness'] < 1.:
            logging.info('Randomly adjust brightness by factor {}'.format(
                options['random_brightness']))
            train_data_processor.process_image(
                tf.image.random_brightness,
                max_delta=options['random_brightness'])

        train_data_processor.process_image(lambda image: image / 255.)
        train_data_processor.process_label(id2trainid_tensor)
        train_data_processor.process_image_and_label(
            random_crop_image_and_label,
            crop_size=tf.constant((options['crop_height'],
                                   options['crop_width'])))

        if options['random_flip_left_right']:
            logging.info('Randomly horizontal flip image and label.')
            train_data_processor.process_image_and_label(
                random_flip_left_right_image_and_label)

        class_weights = np.array(options['class_weights'], dtype=np.float)
        assert class_weights.shape[0] == options['num_classes']
        class_weights = tf.constant(class_weights, dtype=tf.float32)

        # Gets batch iterator.
        # XXX Process validation dataset as well.
        batch = train_data_processor.get_next()

        global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.device('/gpu:0'):
        model = fcn8(options['num_classes'])

        # XXX get proper parameters.
        learning_rate = tf.train.polynomial_decay(
            learning_rate=0.0001,
            global_step=global_step,
            decay_steps=options['max_steps'],
            end_learning_rate=0.000001,
            power=0.9)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)

        trainer = Trainer(model, batch, cross_entropy, class_weights, optimizer, global_step,
                          logdir_fullpath)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if 'vgg_pretrain_ckpt_path' in options.keys():
            model.restore_vgg_weights(sess, options['vgg_pretrain_ckpt_path'])
        trainer.train(sess)

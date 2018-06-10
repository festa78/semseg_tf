#!/usr/bin/python3 -B
"""The script to train cityscapes data by a fully convolutional network (aka FCN).
"""

import argparse
import logging
logging.basicConfig(level=logging.INFO)
import os

import numpy as np
import tensorflow as tf

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
        'tfdata_dir',
        type=str,
        help=
        'Path to the cityscapes tfrecord data directory. The data should be created by sss.data.tfrecord.TFRecordWriter.'
    )
    parser.add_argument(
        'num_classes', type=int, help='The number of classes to use.')
    parser.add_argument(
        '--vgg_pretrain_ckpt_path',
        type=str,
        default=None,
        help='The path to pretrain model of vgg.')
    parser.add_argument(
        '--crop_height',
        type=int,
        default=500,
        help='The crop height size for random cropping.')
    parser.add_argument(
        '--crop_width',
        type=int,
        default=500,
        help='The crop width size for random cropping.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=12,
        help='The batch size on training.')
    parser.add_argument(
        '--max_steps',
        type=int,
        default=15000,
        help='The maximum number of steps to train.')
    parser.add_argument(
        '--no_random_flip_left_right',
        action='store_true',
        default=False,
        help='Do not random flip image horizontally.')
    parser.add_argument(
        '--random_contrast',
        type=float,
        default=.2,
        help='scale factor used to adjust contrast.')
    parser.add_argument(
        '--random_hue',
        type=float,
        default=.2,
        help='scale factor used to adjust hue.')
    parser.add_argument(
        '--random_saturation',
        type=float,
        default=.2,
        help='scale factor used to adjust saturation.')
    parser.add_argument(
        '--random_brightness',
        type=float,
        default=.2,
        help='scale factor used to adjust brightness.')
    parser.add_argument(
        '--num_parallel_calls',
        type=int,
        default=10,
        help='The number of workers for tf.data.Dataset.map().')
    parser.add_argument(
        '--shuffle_buffer_size',
        type=int,
        default=100,
        help='The parameter for tf.data.Dataset.shuffle().')
    parser.add_argument(
        '--prefetch_buffer_size',
        type=int,
        default=100,
        help='The parameter for tf.data.Dataset.prefetch().')
    parser.add_argument(
        '--logdir',
        type=str,
        default='./',
        help='The directory path to save TensorBoard summaries.')

    options = parser.parse_args()

    # Initializations.
    tf.set_random_seed(1234)
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    logdir_fullpath = os.path.expanduser(options.logdir)
    if not os.path.isdir(logdir_fullpath):
        raise AttributeError('--logdir should be existing directory path.')

    # Data part should live in cpu.
    with tf.device('/cpu:0'):
        # Read from tfrecord format data made by sss.data.tfrecord.TFRecordWriter.
        train_dataset = read_tfrecord(
            os.path.join(options.tfdata_dir, 'train.tfrecord'))
        train_data_processor = DataPreprocessor(
            dataset=train_dataset,
            num_parallel_calls=options.num_parallel_calls,
            batch_size=options.batch_size,
            max_epochs=None,
            shuffle_buffer_size=options.shuffle_buffer_size,
            prefetch_buffer_size=options.prefetch_buffer_size)

        val_dataset = read_tfrecord(
            os.path.join(options.tfdata_dir, 'val.tfrecord'))
        val_data_processor = DataPreprocessor(
            dataset=val_dataset,
            num_parallel_calls=10,
            batch_size=1,
            max_epochs=1,
            shuffle_buffer_size=None,
            prefetch_buffer_size=None)

        # Add more pre-procesing blocks.
        if options.random_saturation > 0. and options.random_saturation < 1.:
            logging.info('Randomly adjust saturation by factor {}'.format(
                options.random_saturation))
            lower = 1. - options.random_saturation
            upper = 1. + options.random_saturation
            train_data_processor.process_image(
                tf.image.random_saturation, lower=lower, upper=upper)
        if options.random_contrast > 0. and options.random_contrast < 1.:
            logging.info('Randomly adjust contrast by factor {}'.format(
                options.random_contrast))
            lower = 1. - options.random_contrast
            upper = 1. + options.random_contrast
            train_data_processor.process_image(
                tf.image.random_contrast, lower=lower, upper=upper)
        if options.random_hue > 0. and options.random_hue < .5:
            logging.info('Randomly adjust hue by factor {}'.format(
                options.random_hue))
            train_data_processor.process_image(
                tf.image.random_hue, max_delta=options.random_hue)
        if options.random_brightness > 0. and options.random_brightness < 1.:
            logging.info('Randomly adjust brightness by factor {}'.format(
                options.random_brightness))
            train_data_processor.process_image(
                tf.image.random_brightness, max_delta=options.random_brightness)

        train_data_processor.process_image(lambda image: image / 255.)
        train_data_processor.process_label(id2trainid_tensor)
        train_data_processor.process_image_and_label(
            random_crop_image_and_label,
            crop_size=tf.constant((options.crop_height, options.crop_width)))
        if not options.no_random_flip_left_right:
            logging.info('Randomly horizontal flip image and label.')
            train_data_processor.process_image_and_label(
                random_flip_left_right_image_and_label)

        # Gets batch iterator.
        # XXX Process validation dataset as well.
        batch = train_data_processor.get_next()

        global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.device('/gpu:0'):
        model = fcn8(options.num_classes)

        # XXX get proper parameters.
        learning_rate = tf.train.polynomial_decay(
            learning_rate=0.0001,
            global_step=global_step,
            decay_steps=options.max_steps,
            end_learning_rate=0.000001,
            power=0.9)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(
        #     learning_rate=learning_rate)

        trainer = Trainer(model, batch, cross_entropy, optimizer, global_step,
                          logdir_fullpath)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        model.restore_vgg_weights(sess, options.vgg_pretrain_ckpt_path)
        trainer.train(sess)

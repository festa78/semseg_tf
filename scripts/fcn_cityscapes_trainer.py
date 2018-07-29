#!/usr/bin/python3 -B
"""The script to train cityscapes data by a fully convolutional network (aka FCN).
"""

import argparse
from datetime import datetime
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
from sss.utils.image_processing import random_crop_image_and_label, \
    random_flip_left_right_image_and_label, resize_image_and_label
from sss.utils.losses import cross_entropy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "The script to train cityscapes data by a fully convolutional network (aka FCN)."
    )
    parser.add_argument(
        'params_yaml',
        type=str,
        help=
        'Path to the parameter setting yaml file. Examples in scripts/params')

    args = parser.parse_args()
    options = yaml.load(open(args.params_yaml))
    print(yaml.dump(options))

    # Initializations.
    tf.set_random_seed(1234)
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    save_dir_fullpath = os.path.expanduser(options['save_dir'])
    if not os.path.isdir(save_dir_fullpath):
        raise AttributeError('--save_dir should be existing directory path.')
    save_dir_fullpath = os.path.join(save_dir_fullpath, str(datetime.now()))
    os.makedirs(save_dir_fullpath)
    logging.info('Created save directory to {}'.format(save_dir_fullpath))

    resume_fullpath = None
    if options['resume_path']:
        resume_fullpath = os.path.expanduser(options['resume_path'])

    # Data part should live in cpu.
    with tf.device('/cpu:0'):
        # Read from tfrecord format data made by sss.data.tfrecord.TFRecordWriter.
        train_dataset = read_tfrecord(
            os.path.join(options['tfdata_dir'], 'train_*.tfrecord'))
        train_data_processor = DataPreprocessor(
            dataset=train_dataset,
            num_parallel_calls=options['num_parallel_calls'],
            batch_size=options['batch_size'],
            shuffle_buffer_size=options['shuffle_buffer_size'],
            prefetch_buffer_size=options['prefetch_buffer_size'])

        val_dataset = read_tfrecord(
            os.path.join(options['tfdata_dir'], 'val_*.tfrecord'))
        val_data_processor = DataPreprocessor(
            dataset=val_dataset,
            num_parallel_calls=options['num_parallel_calls'],
            batch_size=1,
            shuffle_buffer_size=1,
            prefetch_buffer_size=1)

        # Pre-process training data.
        # Add more pre-procesing blocks.
        logging.info('Preprocess train data')
        if options['train_resized_height'] is not None and options['train_resized_width'] is not None:
            logging.info('Resize image to ({}, {})'.format(
                options['train_resized_height'],
                options['train_resized_width']))
            train_data_processor.process_image_and_label(
                resize_image_and_label,
                size=tf.constant((options['train_resized_height'],
                                  options['train_resized_width'])))
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

        train_data_processor.process_image_and_label(
            random_crop_image_and_label,
            crop_size=(options['train_crop_height'],
                       options['train_crop_width']))
        train_data_processor.process_image(lambda image: image / 255.)
        train_data_processor.process_label(id2trainid_tensor)

        if options['random_flip_left_right']:
            logging.info('Randomly horizontal flip image and label.')
            train_data_processor.process_image_and_label(
                random_flip_left_right_image_and_label)

        train_iterator = train_data_processor.get_iterator()

        # Pre-process validation data.
        logging.info('Preprocess validation data')
        logging.info('Resize validation image to ({}, {})'.format(
            options['val_resized_height'], options['val_resized_width']))
        val_data_processor.process_image_and_label(
            resize_image_and_label,
            size=tf.constant((options['val_resized_height'],
                              options['val_resized_width'])))
        val_data_processor.process_image(lambda image: image / 255.)
        val_data_processor.process_label(id2trainid_tensor)
        val_iterator = val_data_processor.get_iterator()

        # Get classes weights to use on loss computation.
        train_class_weights = np.array(
            options['train_class_weights'], dtype=np.float)
        val_class_weights = np.array(
            options['val_class_weights'], dtype=np.float)
        assert train_class_weights.shape[0] == options['num_classes']
        assert val_class_weights.shape[0] == options['num_classes']
        train_class_weights = tf.constant(train_class_weights, dtype=tf.float32)
        val_class_weights = tf.constant(val_class_weights, dtype=tf.float32)

        global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.device('/gpu:0'):
        model = fcn8(options['num_classes'])

        # XXX get proper parameters.
        learning_rate = tf.train.polynomial_decay(
            learning_rate=0.001,
            global_step=global_step,
            decay_steps=10000,
            end_learning_rate=0.000001,
            power=0.9)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(
        #     learning_rate=learning_rate)

        trainer = Trainer(
            model=model,
            num_classes=options['num_classes'],
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            loss_fn=cross_entropy,
            optimizer=optimizer,
            global_step=global_step,
            save_dir=save_dir_fullpath,
            resume_path=resume_fullpath,
            train_class_weights=train_class_weights,
            val_class_weights=val_class_weights,
            num_epochs=options['num_epochs'],
            evaluate_freq=options['evaluate_freq'])

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if options['vgg_pretrain_ckpt_path'] and not resume_fullpath:
            model.restore_vgg_weights(sess, options['vgg_pretrain_ckpt_path'],
                                      'model/')
            logging.info('The vgg weights loaded from {}.'.format(options['vgg_pretrain_ckpt_path']))
        trainer.train(sess)

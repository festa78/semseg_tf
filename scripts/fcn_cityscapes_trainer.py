#!/usr/bin/python3 -B
"""The script to train cityscapes data by a fully convolutional network (aka FCN).
"""

import argparse
import os

import numpy as np
import tensorflow as tf

import project_root

from sss.data.cityscapes import id2trainid_tensor
from sss.data.data_preprocessor import DataPreprocessor
from sss.data.tfrecord import read_tfrecord
from sss.models.fcn import fcn8, fcn16, fcn32
from sss.pipelines.trainer import Trainer
from sss.utils.image_processing import random_crop_image_and_label
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
        default=10,
        help='The maximum number of epochs to train.')
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=100,
        help='The maximum number of epochs to train.')
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
            max_epochs=options.max_epochs,
            shuffle_buffer_size=options.shuffle_buffer_size,
            prefetch_buffer_size=options.prefetch_buffer_size)

        val_dataset = read_tfrecord(
            os.path.join(options.tfdata_dir, 'val.tfrecord'))
        val_data_processor = DataPreprocessor(
            dataset=val_dataset,
            num_parallel_calls=10,
            batch_size=1,
            max_epochs=None,
            shuffle_buffer_size=None,
            prefetch_buffer_size=None)

        # Add more pre-procesing blocks.
        train_data_processor.process_label((id2trainid_tensor,))
        train_data_processor.process_image_and_label(
            (random_crop_image_and_label,),
            crop_size=tf.constant((options.crop_height, options.crop_width)))

        # Gets batch iterator.
        batch = train_data_processor.get_next()

        global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.device('/gpu:0'):
        model = fcn32(options.num_classes)

        # XXX get proper parameters.
        optimizer = tf.train.AdamOptimizer()
        # learning_rate = tf.train.polynomial_decay(
        #     learning_rate=0.02,
        #     global_step=global_step,
        #     decay_steps=10000,
        #     end_learning_rate=0.0001,
        #     power=0.9)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02)

        trainer = Trainer(model, batch, cross_entropy, optimizer, global_step,
                          logdir_fullpath)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        model.restore_vgg_weights(sess, options.vgg_pretrain_ckpt_path)
        trainer.train(sess)

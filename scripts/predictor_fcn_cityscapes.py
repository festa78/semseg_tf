#!/usr/bin/python3 -B
"""The script to predict cityscapes data by a fully convolutional network (aka FCN).
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

from src.data.cityscapes import id2trainid_tensor, trainid2color_tensor
from src.data.data_preprocessor import DataPreprocessor
from src.data.tfrecord import read_tfrecord
from src.models.fcn import fcn8, fcn16, fcn32
from src.pipelines.predictor import Predictor
from src.utils.image_processing import resize_image_and_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "The script to predict cityscapes data by a fully convolutional network (aka FCN)."
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

    ckpt_fullpath = os.path.expanduser(options['ckpt_path'])

    # Data part should live in cpu.
    with tf.device('/cpu:0'):
        # Read from tfrecord format data made by src.data.tfrecord.TFRecordWriter.
        test_dataset = read_tfrecord(
            os.path.join(options['tfdata_dir'], 'test_*.tfrecord'))
        test_data_processor = DataPreprocessor(
            dataset=test_dataset,
            num_parallel_calls=options['num_parallel_calls'],
            batch_size=options['batch_size'],
            shuffle_buffer_size=None,
            prefetch_buffer_size=1)

        # Pre-process test data.
        # Add more pre-procesing blocks.
        logging.info('Preprocess test data')
        if options['test_resized_height'] is not None and options['test_resized_width'] is not None:
            logging.info('Resize image to ({}, {})'.format(
                options['test_resized_height'], options['test_resized_width']))
            test_data_processor.process_image_and_label(
                resize_image_and_label,
                size=tf.constant((options['test_resized_height'],
                                  options['test_resized_width'])))

        test_data_processor.process_image(lambda image: image / 255.)
        test_data_processor.process_label(id2trainid_tensor)

        test_iterator = test_data_processor.get_iterator()

    with tf.device('/gpu:0'):
        if options['mode'] == 'fcn32':
            model = fcn32(options['num_classes'])
        elif options['mode'] == 'fcn16':
            model = fcn16(options['num_classes'])
        elif options['mode'] == 'fcn8':
            model = fcn8(options['num_classes'])
        else:
            raise AttributeError('mode {} does not exist.'.format(
                options['mode']))

        predictor = Predictor(
            model=model,
            num_classes=options['num_classes'],
            test_iterator=test_iterator,
            ckpt_path=ckpt_fullpath,
            color_map_fn=trainid2color_tensor,
            save_dir=save_dir_fullpath)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        predictor.prediction(sess)

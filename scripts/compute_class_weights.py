#!/usr/bin/python3 -B
"""The script to compute class distribution.
"""

import argparse
import logging
logging.basicConfig(level=logging.INFO)
import os

import numpy as np
import tensorflow as tf

import project_root

from sss.data.cityscapes import id2label
from sss.data.tfrecord import read_tfrecord

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="The script to compute class distribution.")
    parser.add_argument(
        'data_path', type=str, help='Path to the tfrecord data path.')
    parser.add_argument('num_classes', type=int, help='Number of classes.')

    args = parser.parse_args()

    # Read from tfrecord format data made by sss.data.tfrecord.TFRecordWriter.
    train_dataset = read_tfrecord(args.data_path)
    next_element = train_dataset.make_one_shot_iterator().get_next()

    # Count the number of classes.
    class_counts = np.zeros((args.num_classes, 1), dtype=np.int)
    with tf.Session() as sess:
        loop = 0
        while True:
            if loop % 100 == 0:
                print(loop)
            try:
                label = sess.run(next_element['label'])
                label = np.array(
                    [id2label[i].trainId for i in label.flatten()]).reshape(
                        label.shape)
                for i, count in enumerate(np.bincount(label.flatten())):
                    class_counts[i] += count
                loop += 1
                # if loop > 10:
                #     break
            except tf.errors.OutOfRangeError:
                break

    class_weights = class_counts / np.sum(class_counts)
    class_inv_weights = np.sum(class_counts) / (class_counts + 1e-9)
    class_inv_weights /= np.sum(class_inv_weights)

    print('counts: ', class_counts)
    print('weights: ', class_weights)
    print('inv_weights: ', class_inv_weights)

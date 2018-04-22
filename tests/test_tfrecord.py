"""Test set for TFRecord wrapper classes.
"""

import argparse
import os

from PIL import Image
import numpy as np
import pytest
import tensorflow as tf

import project_root

from sss.tools.tfrecord.cityscapes import CityscapesTFRecordWriter, CityscapesTFRecordReader


def _create_sample_cityscapes_structure(tmpdir):
    """Creates dummy cityscapes like data structure.

    Returns
    -------
    root_dir_path : str
        Root path to the created data structure.
    data_list : dict
        Dummy data dictionary contains 'image_list' and 'label_list'.
    """
    # Constants.
    ROOTS = ['leftImg8bit', 'gtFine']
    SUBDIRS = ['aaa', 'bbb']
    DATA_CATEGORY = ['train', 'val', 'test']
    FILENAMES = ['test1', 'test2']
    IMAGE_WIDTH, IMAGE_HEIGHT = 100, 100

    np.random.seed(1234)

    data_list = {}
    for root in ROOTS:
        tmpdir.mkdir(root)
    for cat in DATA_CATEGORY:
        image_list = []
        label_list = []
        for root in ROOTS:
            tmpdir.mkdir(root, cat)
            for sub in SUBDIRS:
                tmpdir.mkdir(root, cat, sub)
                for filename in FILENAMES:
                    # Creates empty files.
                    if root == ROOTS[0]:
                        path = tmpdir.join(root, cat, sub, filename + '.png')
                        image = np.random.randint(255, size=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
                        image = Image.fromarray(image.astype(np.uint8))
                        # Convert path from py.path.local to str.
                        image.save(str(path))
                        image_list.append(path)
                    else:
                        path = tmpdir.join(root, cat, sub,
                                           filename + '_labelIds.png')
                        label = np.random.randint(10, size=(IMAGE_WIDTH, IMAGE_HEIGHT))
                        label = Image.fromarray(label.astype(np.uint8))
                        # Convert path from py.path.local to str.
                        label.save(str(path))
                        label_list.append(path)
        data_list[cat] = {'image_list': image_list, 'label_list': label_list}
    root_dir_path = tmpdir.join()
    return root_dir_path, data_list


def test_cityscapes_get_file_path(tmpdir):
    """Test it can get file paths from cityscapes like data structure.
    """
    input_dir, gt_data_list = _create_sample_cityscapes_structure(
        tmpdir)
    output_dir = input_dir
    writer = CityscapesTFRecordWriter(input_dir, output_dir)
    dut = CityscapesTFRecordWriter(input_dir, output_dir)
    dut.get_file_path()
    for cat1, cat2 in zip(dut.data_list.values(), gt_data_list.values()):
        for list1, list2 in zip(cat1.values(), cat2.values()):
            # Not care about orders.
            assert set(list1) == set(list2)


def test_read_tfrecord(tmpdir):
    """Test it can read the tfrecord file correctly.
    """
    # Constants.
    DATA_CATEGORY = ['train', 'val', 'test']

    # Make a dummy tfrecord file.
    input_dir, gt_data_list = _create_sample_cityscapes_structure(
        tmpdir)
    # Convert from py.path.local to str.
    output_dir = input_dir
    writer = CityscapesTFRecordWriter(input_dir, output_dir)
    writer.run()

    # Read the created tfrecord file.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        for category in DATA_CATEGORY:
            dut = CityscapesTFRecordReader(os.path.join(output_dir, category + '.tfrecord'), sess)
            # The op for initializing the variables.
            sess.run(init_op)
            for i, (image, label, filename) in enumerate(dut):
                gt_image = np.array(Image.open(open(filename, 'rb')).convert('RGB'))
                assert np.all(np.equal(image, gt_image))

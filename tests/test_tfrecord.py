"""Test set for TFRecord wrapper classes.
"""

import os

from PIL import Image
import numpy as np
import tensorflow as tf

import project_root

from sss.data.tfrecord import read_tfrecord, write_tfrecord
from sss.data.cityscapes import get_file_path

IMAGE_WIDTH, IMAGE_HEIGHT = 100, 100


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
                        image = np.random.randint(
                            255, size=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
                        image = Image.fromarray(image.astype(np.uint8))
                        # Convert path from py.path.local to str.
                        image.save(str(path))
                        image_list.append(path)
                    else:
                        path = tmpdir.join(root, cat, sub,
                                           filename + '_labelIds.png')
                        label = np.random.randint(
                            10, size=(IMAGE_WIDTH, IMAGE_HEIGHT))
                        label = Image.fromarray(label.astype(np.uint8))
                        # Convert path from py.path.local to str.
                        label.save(str(path))
                        label_list.append(path)
        data_list[cat] = {'image_list': image_list, 'label_list': label_list}
    root_dir_path = tmpdir.join()
    return root_dir_path, data_list


def test_write_read_tfrecord(tmpdir):
    """Test it can write and read the tfrecord file correctly.
    """
    # Constants.
    DATA_CATEGORY = ['train', 'val', 'test']

    # Make a dummy tfrecord file.
    # XXX use more simple structure.
    input_dir, gt_data_list = _create_sample_cityscapes_structure(tmpdir)
    output_dir = input_dir
    # Convert from py.path.local to str.
    data_list = get_file_path(input_dir)
    write_tfrecord(data_list, output_dir)

    # Read the created tfrecord file.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    for category in DATA_CATEGORY:
        dataset = read_tfrecord(
            os.path.join(output_dir, category + '_0000.tfrecord'))
        next_element = dataset.make_one_shot_iterator().get_next()
        with tf.Session() as sess:
            # The op for initializing the variables.
            sess.run(init_op)
            i = 0
            while True:
                try:
                    sample = sess.run(next_element)
                    gt_image = np.array(
                        Image.open(open(sample['filename'].decode(),
                                        'rb')).convert('RGB'))
                    assert np.array_equal(sample['image'], gt_image)
                    assert sample['height'] == IMAGE_HEIGHT
                    assert sample['width'] == IMAGE_WIDTH
                    i += 1
                except tf.errors.OutOfRangeError:
                    assert i == 4
                    break

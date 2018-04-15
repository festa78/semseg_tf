"""Test set for TFRecord wrapper classes.
"""

import argparse
import os

import pytest

import project_root

from tools.tfrecord.cityscapes import CityscapesTFRecordWriter


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
                        open(path, 'a').close()
                        image_list.append(path)
                    else:
                        path = tmpdir.join(root, cat, sub,
                                           filename + '_labelIds.png')
                        open(path, 'a').close()
                        label_list.append(path)
        data_list[cat] = {'image_list': image_list, 'label_list': label_list}
    root_dir_path = tmpdir.join()
    return root_dir_path, data_list


def test_cityscapes_get_file_path(tmpdir):
    """Test it can get file paths from cityscapes like data structure.
    """
    dummy_options = argparse.Namespace()
    dummy_options.input_dir, gt_data_list = _create_sample_cityscapes_structure(
        tmpdir)
    dummy_options.output_dir = os.path.join(dummy_options.input_dir)
    writer = CityscapesTFRecordWriter(dummy_options)
    writer.get_file_path()
    for cat1, cat2 in zip(writer.data_list.values(), gt_data_list.values()):
        for list1, list2 in zip(cat1.values(), cat2.values()):
            # Not care about orders.
            assert set(list1) == set(list2)

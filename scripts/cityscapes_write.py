#!/usr/bin/python3 -B
"""The script to make TFRecord datasets from {image, label} pair file paths.
"""

import argparse

import project_root

from sss.data.cityscapes import get_cityscapes_file_path
from sss.data.tfrecord import write_tfrecord

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "The script to make TFRecord datasets from {image, label} pair file paths."
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help=
        'Path to the cityscapes data directory. The directory structure is assumed to be default.'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Path to directory to save the created .tfrecord data.')
    options = parser.parse_args()

    data_list = get_cityscapes_file_path(options.input_dir)
    write_tfrecord(data_list, options.output_dir)

#!/usr/bin/python3 -B

import argparse

import project_root

from sss.tools.tfrecord.cityscapes import CityscapesTFRecordWriter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Make TFRecord datasets from {image, label} pair file paths."
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

    writer = CityscapesTFRecordWriter(options.input_dir, options.output_dir)
    writer.run()

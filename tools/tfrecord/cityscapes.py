#!/usr/bin/python3 -B

import argparse
import os

import glob
import tensorflow as tf

import project_root

import tools.tfrecord.base as base


class CityscapesTFRecordWriter(base.BaseTFRecordWriter):
    """Make TFRecord datasets from row Cityscapes data.
    """
    # Constants.
    IMAGE_ROOT = 'leftImg8bit'
    LABEL_ROOT = 'gtFine'
    LABEL_SURFIX = '_labelIds.png'
    DATA_CATEGORY = ['train', 'val', 'test']

    def __init__(self, options):
        # XXX: Need some assertions to check that input files properly exists.
        self.input_dir = os.path.abspath(os.path.expanduser(options.input_dir))
        self.output_dir = os.path.abspath(os.path.expanduser(options.output_dir))
        super().__init__()

    def get_file_path(self):
        """Cityscapes specific way to get file paths.
        """
        for category in self.DATA_CATEGORY:
            image_list = glob.glob(
                os.path.join(self.input_dir, self.IMAGE_ROOT, category,
                             '*/*'))

            # Get label path corresponds to each image path.
            label_list = []
            for f in image_list:
                root_path = os.path.join(self.input_dir,
                                         self.LABEL_ROOT, category)
                area_name = f.split('/')[-2]
                base_name = os.path.basename(f).replace(
                    self.IMAGE_ROOT, self.LABEL_ROOT).replace('.png', self.LABEL_SURFIX)
                label_list.append(os.path.join(root_path, area_name, base_name))

            self.data_list[category] = {
                'image_list': image_list,
                'label_list': label_list
            }

    def run(self):
        # Get {image, label} pair file path.
        self.get_file_path()

        # Write tfrecord file.
        self._write_tfrecord(self.output_dir)


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

    writer = CityscapesTFRecordWriter(options)
    writer.run()
